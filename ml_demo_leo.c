#include "ml_demo_leo.h"
#include "accelerometer.h"
#include "sl_status.h"
#include "sl_iostream.h"
#include "sl_sleeptimer.h"
#include "sl_board_control.h"
#include "sl_i2cspm_instances.h"
#include "sl_si7210.h"
#include <stdbool.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HALL_MEASUREMENT_SCALE_UT 20000U

static ml_demo_leo_linear_model_t g_model;
static imu_data_t g_window[ML_DEMO_LEO_WINDOW_SIZE];
static int g_window_count = 0;
static uint32_t g_sample_count = 0;
static uint32_t g_window_log_count = 0;
static const uint32_t ML_DEMO_LEO_LOG_EVERY_WINDOWS = 5U;
static uint32_t g_training_windows_remaining = 0U;
static char g_command_buffer[24];
static size_t g_command_length = 0U;
#define ML_DEMO_LEO_REPLAY_CAPACITY 32U
static float g_replay_features[2][ML_DEMO_LEO_REPLAY_CAPACITY][ML_DEMO_LEO_FEATURE_COUNT];
static uint32_t g_replay_count[2] = { 0U, 0U };
static uint32_t g_replay_cursor[2] = { 0U, 0U };
static ml_demo_leo_mode_t g_mode = ML_DEMO_LEO_MODE_LOCAL_TRAINING;
static ml_demo_leo_model_type_t g_model_type = ML_DEMO_LEO_MODEL_LINEAR;
static bool g_federated_enabled = false;
static bool g_sensor_log_enabled = false;
static bool g_hall_ready = false;
static uint32_t g_last_sensor_log_ms = 0U;
static int g_manual_target = 0;

static uint32_t model_parameter_count(void);
static bool model_set_parameter(uint32_t index, float value);
static float model_get_parameter(uint32_t index);
static void serial_dump_model(void);

static void print_command_menu(void)
{
  printf("commands: p N=train parado, m N=train mexendo, i=inference\n");
  printf("commands: model linear|cnn, f on|off, dump, w I VALUE\n");
  printf("commands: sensors on|off (print accel+gyro+Hall every 2 seconds)\n");
  printf("N = janelas de 32 amostras de acelerometro e giroscopio (~1.28 s)\n");
  printf("> ");
}

static void execute_command(void)
{
  char *end = NULL;
  unsigned long window_count;

  if (g_command_length == 0U) {
    return;
  }

  g_command_buffer[g_command_length] = '\0';

  if (strcmp(g_command_buffer, "model linear") == 0) {
    g_model_type = ML_DEMO_LEO_MODEL_LINEAR;
    printf("model selected: linear\n");
    return;
  }

  if (strcmp(g_command_buffer, "model cnn") == 0) {
    g_model_type = ML_DEMO_LEO_MODEL_CNN;
    printf("model selected: cnn1d filters=%d kernel=%d\n",
           ML_DEMO_LEO_CNN_FILTERS, ML_DEMO_LEO_CNN_KERNEL);
    return;
  }

  if (strcmp(g_command_buffer, "f on") == 0) {
    g_federated_enabled = true;
    printf("federated: enabled (local training remains on this EFR32)\n");
    return;
  }

  if (strcmp(g_command_buffer, "f off") == 0) {
    g_federated_enabled = false;
    printf("federated: disabled\n");
    return;
  }

  if (strcmp(g_command_buffer, "dump") == 0) {
    serial_dump_model();
    return;
  }

  if (strcmp(g_command_buffer, "sensors on") == 0) {
    g_sensor_log_enabled = true;
    g_last_sensor_log_ms = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());
    printf("sensor logging enabled (every 2 seconds)\n");
    return;
  }

  if (strcmp(g_command_buffer, "sensors off") == 0) {
    g_sensor_log_enabled = false;
    printf("sensor logging disabled\n");
    return;
  }

  if (g_command_buffer[0] == 'w' && g_command_buffer[1] == ' ') {
    char *value_start = NULL;
    unsigned long index = strtoul(&g_command_buffer[2], &value_start, 10);
    float value;

    if (value_start == NULL || *value_start != ' ') {
      printf("invalid weight command; use w I VALUE\n");
      return;
    }

    value = strtof(value_start + 1, &end);
    if (*end != '\0' || index >= model_parameter_count()) {
      printf("invalid weight index/value\n");
      return;
    }

    if (model_set_parameter((uint32_t)index, value)) {
      printf("weight[%lu]=%.7f\n", index, value);
    }
    return;
  }

  if (g_command_buffer[0] == 'i' && g_command_length == 1U) {
    g_mode = ML_DEMO_LEO_MODE_INFERENCE;
    g_training_windows_remaining = 0U;
    printf("command: inference started\n");
    return;
  }

  if ((g_command_buffer[0] != 'p' && g_command_buffer[0] != 'm')
      || g_command_length < 3U
      || g_command_buffer[1] != ' ') {
    printf("invalid command\n");
    print_command_menu();
    return;
  }

  window_count = strtoul(&g_command_buffer[2], &end, 10);
  if (window_count == 0UL || *end != '\0' || window_count > UINT32_MAX) {
    printf("invalid window count\n");
    print_command_menu();
    return;
  }

  g_mode = ML_DEMO_LEO_MODE_LOCAL_TRAINING;
  g_manual_target = (g_command_buffer[0] == 'm') ? 1 : 0;
  g_training_windows_remaining = (uint32_t)window_count;

  printf("command: training class=%d (%s), windows=%lu, samples=%lu, approx_seconds=%lu\n",
         g_manual_target,
         (g_manual_target == 0) ? "parado" : "mexendo",
         window_count,
         window_count * ML_DEMO_LEO_WINDOW_SIZE,
         (window_count * ML_DEMO_LEO_WINDOW_SIZE) / 25UL);
}

static void poll_serial_command(void)
{
  char received;

  while (sl_iostream_getchar(SL_IOSTREAM_STDIN, &received) == SL_STATUS_OK) {
    if (received == '\r' || received == '\n') {
      execute_command();
      g_command_length = 0U;
    } else if (g_command_length < sizeof(g_command_buffer) - 1U) {
      g_command_buffer[g_command_length++] = received;
    }
  }
}

static float sigmoidf(float x)
{
  if (x >= 0.0f) {
    return 1.0f / (1.0f + expf(-x));
  }

  float e = expf(x);
  return e / (1.0f + e);
}

static float imu_channel(const imu_data_t *sample, int channel)
{
  switch (channel) {
    case 0: return sample->x;
    case 1: return sample->y;
    case 2: return sample->z;
    case 3: return sample->gx;
    case 4: return sample->gy;
    default: return sample->gz;
  }
}

static void compute_sensor_features(const imu_data_t *window,
                                    int count,
                                    int first_channel,
                                    float out[ML_DEMO_LEO_FEATURES_PER_SENSOR])
{
  float sum_mag = 0.0f;
  float sum_mag_sq = 0.0f;
  float mean_delta = 0.0f;
  float delta_sq = 0.0f;
  float sum_axis[3] = { 0.0f, 0.0f, 0.0f };
  float sum_axis_sq[3] = { 0.0f, 0.0f, 0.0f };

  for (int i = 0; i < count; ++i) {
    float axis[3] = {
      imu_channel(&window[i], first_channel),
      imu_channel(&window[i], first_channel + 1),
      imu_channel(&window[i], first_channel + 2)
    };
    float mag = sqrtf(axis[0] * axis[0] +
                      axis[1] * axis[1] +
                      axis[2] * axis[2]);
    sum_mag += mag;
    sum_mag_sq += mag * mag;
    for (int axis_index = 0; axis_index < 3; ++axis_index) {
      sum_axis[axis_index] += axis[axis_index];
      sum_axis_sq[axis_index] += axis[axis_index] * axis[axis_index];
    }

    if (i > 0) {
      float dx = axis[0] - imu_channel(&window[i - 1], first_channel);
      float dy = axis[1] - imu_channel(&window[i - 1], first_channel + 1);
      float dz = axis[2] - imu_channel(&window[i - 1], first_channel + 2);
      float delta = sqrtf(dx * dx + dy * dy + dz * dz);
      mean_delta += delta;
      delta_sq += delta * delta;
    }
  }

  float inv_n = 1.0f / (float)count;
  float inv_delta_n = 1.0f / (float)(count - 1);
  float mean_mag = sum_mag * inv_n;
  float mean_delta_value = mean_delta * inv_delta_n;

  out[0] = mean_mag * 0.001f;
  out[1] = sqrtf(fmaxf(0.0f, sum_mag_sq * inv_n - mean_mag * mean_mag)) * 0.001f;
  out[2] = mean_delta_value * 0.001f;
  out[3] = sqrtf(fmaxf(0.0f, delta_sq * inv_delta_n - mean_delta_value * mean_delta_value)) * 0.001f;
  for (int axis_index = 0; axis_index < 3; ++axis_index) {
    float mean_axis = sum_axis[axis_index] * inv_n;
    out[4 + axis_index] = sqrtf(fmaxf(0.0f,
                                      sum_axis_sq[axis_index] * inv_n
                                      - mean_axis * mean_axis)) * 0.001f;
  }
}

static void compute_features(const imu_data_t *window,
                             int count,
                             float out[ML_DEMO_LEO_FEATURE_COUNT])
{
  compute_sensor_features(window, count, 0, &out[0]);
  compute_sensor_features(window, count, 3,
                          &out[ML_DEMO_LEO_FEATURES_PER_SENSOR]);
}

static void model_init(ml_demo_leo_linear_model_t *model)
{
  for (int i = 0; i < ML_DEMO_LEO_FEATURE_COUNT; ++i) {
    model->weights[i] = 0.01f;
  }
  model->bias = 0.0f;

  for (int filter = 0; filter < ML_DEMO_LEO_CNN_FILTERS; ++filter) {
    model->cnn_bias[filter] = 0.0f;
    for (int kernel = 0; kernel < ML_DEMO_LEO_CNN_KERNEL; ++kernel) {
      for (int channel = 0; channel < ML_DEMO_LEO_CNN_CHANNELS; ++channel) {
        model->cnn_weights[filter][kernel][channel] = 0.01f;
      }
    }
    model->cnn_output_weights[filter] = 0.01f;
  }
  model->cnn_output_bias = 0.0f;
}

static float cnn_predict(const ml_demo_leo_linear_model_t *model,
                         const imu_data_t *window,
                         float hidden[ML_DEMO_LEO_CNN_FILTERS])
{
  const int positions = ML_DEMO_LEO_WINDOW_SIZE - ML_DEMO_LEO_CNN_KERNEL + 1;
  float score = model->cnn_output_bias;

  for (int filter = 0; filter < ML_DEMO_LEO_CNN_FILTERS; ++filter) {
    float sum = 0.0f;
    for (int position = 0; position < positions; ++position) {
      float convolution = model->cnn_bias[filter];
      for (int kernel = 0; kernel < ML_DEMO_LEO_CNN_KERNEL; ++kernel) {
        const float input[ML_DEMO_LEO_CNN_CHANNELS] = {
          window[position + kernel].x,
          window[position + kernel].y,
          window[position + kernel].z,
          window[position + kernel].gx,
          window[position + kernel].gy,
          window[position + kernel].gz
        };
        for (int channel = 0; channel < ML_DEMO_LEO_CNN_CHANNELS; ++channel) {
          convolution += model->cnn_weights[filter][kernel][channel]
                         * input[channel] * 0.001f;
        }
      }
      sum += fmaxf(0.0f, convolution);
    }
    hidden[filter] = sum / (float)positions;
    score += model->cnn_output_weights[filter] * hidden[filter];
  }

  return sigmoidf(score);
}

static float model_predict(const ml_demo_leo_linear_model_t *model,
                           const imu_data_t *window,
                           const float features[ML_DEMO_LEO_FEATURE_COUNT])
{
  if (g_model_type == ML_DEMO_LEO_MODEL_CNN) {
    float hidden[ML_DEMO_LEO_CNN_FILTERS];
    return cnn_predict(model, window, hidden);
  }

  float score = model->bias;
  for (int i = 0; i < ML_DEMO_LEO_FEATURE_COUNT; ++i) {
    score += model->weights[i] * features[i];
  }
  return sigmoidf(score);
}

static void model_train_local(ml_demo_leo_linear_model_t *model,
                             const imu_data_t *window,
                             const float features[ML_DEMO_LEO_FEATURE_COUNT],
                             int target,
                             float learning_rate)
{
  float y = model_predict(model, window, features);
  float error = (float)target - y;

  if (g_model_type == ML_DEMO_LEO_MODEL_CNN) {
    const int positions = ML_DEMO_LEO_WINDOW_SIZE - ML_DEMO_LEO_CNN_KERNEL + 1;
    float hidden[ML_DEMO_LEO_CNN_FILTERS];
    float old_output_weights[ML_DEMO_LEO_CNN_FILTERS];

    cnn_predict(model, window, hidden);
    for (int filter = 0; filter < ML_DEMO_LEO_CNN_FILTERS; ++filter) {
      old_output_weights[filter] = model->cnn_output_weights[filter];
      model->cnn_output_weights[filter] += learning_rate * error * hidden[filter];
    }
    model->cnn_output_bias += learning_rate * error;

    for (int filter = 0; filter < ML_DEMO_LEO_CNN_FILTERS; ++filter) {
      for (int position = 0; position < positions; ++position) {
        float convolution = model->cnn_bias[filter];
        for (int kernel = 0; kernel < ML_DEMO_LEO_CNN_KERNEL; ++kernel) {
          const float input[ML_DEMO_LEO_CNN_CHANNELS] = {
            window[position + kernel].x,
            window[position + kernel].y,
            window[position + kernel].z,
            window[position + kernel].gx,
            window[position + kernel].gy,
            window[position + kernel].gz
          };
          for (int channel = 0; channel < ML_DEMO_LEO_CNN_CHANNELS; ++channel) {
            convolution += model->cnn_weights[filter][kernel][channel]
                           * input[channel] * 0.001f;
          }
        }

        if (convolution > 0.0f) {
          float gradient = error * old_output_weights[filter] / (float)positions;
          model->cnn_bias[filter] += learning_rate * gradient;
          for (int kernel = 0; kernel < ML_DEMO_LEO_CNN_KERNEL; ++kernel) {
            for (int channel = 0; channel < ML_DEMO_LEO_CNN_CHANNELS; ++channel) {
              model->cnn_weights[filter][kernel][channel]
                += learning_rate * gradient
                   * imu_channel(&window[position + kernel], channel)
                   * 0.001f;
            }
          }
        }
      }
    }
    return;
  }

  for (int i = 0; i < ML_DEMO_LEO_FEATURE_COUNT; ++i) {
    model->weights[i] += learning_rate * error * features[i];
  }
  model->bias += learning_rate * error;
}

static void replay_store(const float features[ML_DEMO_LEO_FEATURE_COUNT], int target)
{
  uint32_t index = g_replay_cursor[target];

  for (int i = 0; i < ML_DEMO_LEO_FEATURE_COUNT; ++i) {
    g_replay_features[target][index][i] = features[i];
  }

  g_replay_cursor[target] = (index + 1U) % ML_DEMO_LEO_REPLAY_CAPACITY;
  if (g_replay_count[target] < ML_DEMO_LEO_REPLAY_CAPACITY) {
    g_replay_count[target]++;
  }
}

static void model_train_with_replay(const float features[ML_DEMO_LEO_FEATURE_COUNT], int target)
{
  int other_target = 1 - target;

  replay_store(features, target);
  model_train_local(&g_model, g_window, features, target, 0.01f);

  if (g_replay_count[other_target] > 0U) {
    uint32_t index = g_replay_cursor[other_target] % g_replay_count[other_target];
    if (g_model_type == ML_DEMO_LEO_MODEL_LINEAR) {
      model_train_local(&g_model, g_window,
                        g_replay_features[other_target][index],
                      other_target,
                      0.01f);
    }
  }
}

static uint32_t model_parameter_count(void)
{
  if (g_model_type == ML_DEMO_LEO_MODEL_LINEAR) {
    return ML_DEMO_LEO_FEATURE_COUNT + 1U;
  }
  return (ML_DEMO_LEO_CNN_FILTERS * ML_DEMO_LEO_CNN_KERNEL
          * ML_DEMO_LEO_CNN_CHANNELS)
         + ML_DEMO_LEO_CNN_FILTERS + ML_DEMO_LEO_CNN_FILTERS + 1U;
}

static float model_get_parameter(uint32_t index)
{
  if (g_model_type == ML_DEMO_LEO_MODEL_LINEAR) {
    return (index < ML_DEMO_LEO_FEATURE_COUNT)
           ? g_model.weights[index] : g_model.bias;
  }

  uint32_t convolution_count = ML_DEMO_LEO_CNN_FILTERS
                               * ML_DEMO_LEO_CNN_KERNEL
                               * ML_DEMO_LEO_CNN_CHANNELS;
  if (index < convolution_count) {
    uint32_t filter = index / (ML_DEMO_LEO_CNN_KERNEL * ML_DEMO_LEO_CNN_CHANNELS);
    uint32_t remainder = index % (ML_DEMO_LEO_CNN_KERNEL * ML_DEMO_LEO_CNN_CHANNELS);
    uint32_t kernel = remainder / ML_DEMO_LEO_CNN_CHANNELS;
    uint32_t channel = remainder % ML_DEMO_LEO_CNN_CHANNELS;
    return g_model.cnn_weights[filter][kernel][channel];
  }
  index -= convolution_count;
  if (index < ML_DEMO_LEO_CNN_FILTERS) {
    return g_model.cnn_bias[index];
  }
  index -= ML_DEMO_LEO_CNN_FILTERS;
  if (index < ML_DEMO_LEO_CNN_FILTERS) {
    return g_model.cnn_output_weights[index];
  }
  return g_model.cnn_output_bias;
}

static bool model_set_parameter(uint32_t index, float value)
{
  if (index >= model_parameter_count()) {
    return false;
  }

  if (g_model_type == ML_DEMO_LEO_MODEL_LINEAR) {
    if (index < ML_DEMO_LEO_FEATURE_COUNT) {
      g_model.weights[index] = value;
    } else {
      g_model.bias = value;
    }
    return true;
  }

  uint32_t convolution_count = ML_DEMO_LEO_CNN_FILTERS
                               * ML_DEMO_LEO_CNN_KERNEL
                               * ML_DEMO_LEO_CNN_CHANNELS;
  if (index < convolution_count) {
    uint32_t filter = index / (ML_DEMO_LEO_CNN_KERNEL * ML_DEMO_LEO_CNN_CHANNELS);
    uint32_t remainder = index % (ML_DEMO_LEO_CNN_KERNEL * ML_DEMO_LEO_CNN_CHANNELS);
    uint32_t kernel = remainder / ML_DEMO_LEO_CNN_CHANNELS;
    uint32_t channel = remainder % ML_DEMO_LEO_CNN_CHANNELS;
    g_model.cnn_weights[filter][kernel][channel] = value;
  } else {
    index -= convolution_count;
    if (index < ML_DEMO_LEO_CNN_FILTERS) {
      g_model.cnn_bias[index] = value;
    } else {
      index -= ML_DEMO_LEO_CNN_FILTERS;
      if (index < ML_DEMO_LEO_CNN_FILTERS) {
        g_model.cnn_output_weights[index] = value;
      } else {
        g_model.cnn_output_bias = value;
      }
    }
  }
  return true;
}

static void serial_dump_model(void)
{
  printf("FL_MODEL_BEGIN type=%s count=%lu federated=%d\n",
         (g_model_type == ML_DEMO_LEO_MODEL_CNN) ? "cnn" : "linear",
         (unsigned long)model_parameter_count(),
         g_federated_enabled ? 1 : 0);
  for (uint32_t index = 0U; index < model_parameter_count(); ++index) {
    printf("w %lu %.9g\n", (unsigned long)index, model_get_parameter(index));
  }
  printf("FL_MODEL_END\n");
}

void ml_demo_leo_set_mode(ml_demo_leo_mode_t mode)
{
  g_mode = mode;
}

void ml_demo_leo_set_manual_target(int target)
{
  if (target == 0 || target == 1) {
    g_manual_target = target;
  }
}

void ml_demo_leo_set_model(ml_demo_leo_model_type_t model_type)
{
  if (model_type == ML_DEMO_LEO_MODEL_LINEAR
      || model_type == ML_DEMO_LEO_MODEL_CNN) {
    g_model_type = model_type;
  }
}

void ml_demo_leo_set_federated(bool enabled)
{
  g_federated_enabled = enabled;
}

void ml_demo_leo_init(void)
{
  sl_status_t setup_status;

  /*
   * The Hall sensor and IMU power-enable lines are shared on the BRD2601B.
   * Power the Hall sensor and let it settle before starting IMU calibration.
   */
  g_hall_ready = false;
  setup_status = sl_board_enable_sensor(SL_BOARD_SENSOR_HALL);
  if (setup_status != SL_STATUS_OK) {
    printf("ml_demo_leo_init: Hall power enable failed (status=%lu)\n",
           (unsigned long)setup_status);
  } else {
    sl_sleeptimer_delay_millisecond(10U);
    setup_status = sl_si7210_init(sl_i2cspm_sensor);
    g_hall_ready = (setup_status == SL_STATUS_OK);
    if (g_hall_ready) {
      printf("ml_demo_leo_init: Si7210 Hall sensor ready\n");
    } else {
      printf("ml_demo_leo_init: Si7210 initialization failed (status=%lu)\n",
             (unsigned long)setup_status);
    }
  }

  setup_status = accelerometer_setup();
  if (setup_status != SL_STATUS_OK) {
    printf("ml_demo_leo_init: accelerometer setup failed\n");
    return;
  }

  while (1) {
    imu_data_t sample;
    sl_status_t status = imu_read(&sample, 1);
    if (status == SL_STATUS_OK) {
      printf("ml_demo_leo_init: IMU ready accel=(%.2f, %.2f, %.2f) gyro=(%.2f, %.2f, %.2f)\n",
             sample.x,
             sample.y,
             sample.z,
             sample.gx,
             sample.gy,
             sample.gz);
      break;
    }
  }

  model_init(&g_model);
  g_window_count = 0;
  g_sample_count = 0;
  g_window_log_count = 0;
  g_training_windows_remaining = 0U;
  g_command_length = 0U;
  g_replay_count[0] = 0U;
  g_replay_count[1] = 0U;
  g_replay_cursor[0] = 0U;
  g_replay_cursor[1] = 0U;
  g_mode = ML_DEMO_LEO_MODE_LOCAL_TRAINING;
  g_model_type = ML_DEMO_LEO_MODEL_LINEAR;
  g_federated_enabled = false;
  g_manual_target = 0;

  printf("ml_demo_leo_init: accel+gyro model ready (linear default)\n");
  printf("ml_demo_leo_init: waiting for serial command\n");
  print_command_menu();
}

void ml_demo_leo_process(void)
{
  imu_data_t sample;
  sl_status_t status = imu_read(&sample, 1);
  uint32_t now_ms;

  poll_serial_command();

  if (status != SL_STATUS_OK) {
    return;
  }

  now_ms = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());
  if (g_sensor_log_enabled && (uint32_t)(now_ms - g_last_sensor_log_ms) >= 2000U) {
    if (g_hall_ready) {
      float hall_field_mT;
      sl_status_t hall_status = sl_si7210_measure(sl_i2cspm_sensor,
                                                  HALL_MEASUREMENT_SCALE_UT,
                                                  &hall_field_mT);
      if (hall_status == SL_STATUS_OK) {
        printf("imu accel=(%.2f, %.2f, %.2f) gyro=(%.2f, %.2f, %.2f) hall=%.3f mT\n",
               sample.x,
               sample.y,
               sample.z,
               sample.gx,
               sample.gy,
               sample.gz,
               hall_field_mT);
      } else {
        printf("Hall measurement failed (status=%lu)\n",
               (unsigned long)hall_status);
      }
    } else {
      printf("imu accel=(%.2f, %.2f, %.2f) gyro=(%.2f, %.2f, %.2f) hall=unavailable\n",
             sample.x,
             sample.y,
             sample.z,
             sample.gx,
             sample.gy,
             sample.gz);
    }
    g_last_sensor_log_ms = now_ms;
  }

  g_window[g_window_count++] = sample;
  g_sample_count++;

  if (g_window_count < ML_DEMO_LEO_WINDOW_SIZE) {
    sl_sleeptimer_delay_millisecond(40U);
    return;
  }

  const bool should_log = ((g_window_log_count % ML_DEMO_LEO_LOG_EVERY_WINDOWS) == 0U);
  g_window_log_count++;

  float features[ML_DEMO_LEO_FEATURE_COUNT];
  compute_features(g_window, ML_DEMO_LEO_WINDOW_SIZE, features);

  if (g_command_length > 0U) {
    execute_command();
    g_command_length = 0U;
  }

  bool training_window = (g_mode == ML_DEMO_LEO_MODE_LOCAL_TRAINING)
                         && (g_training_windows_remaining > 0U);

  if (g_mode == ML_DEMO_LEO_MODE_LOCAL_TRAINING && !training_window) {
    g_window_count = 0;
    sl_sleeptimer_delay_millisecond(40U);
    return;
  }

  float score = model_predict(&g_model, g_window, features);
  int predicted_label = (score > 0.5f) ? 1 : 0;

  if (training_window) {
    model_train_with_replay(features, g_manual_target);
    g_training_windows_remaining--;
  }

  if (!should_log) {
    if (training_window && g_training_windows_remaining == 0U) {
      printf("training complete: class=%d (%s)\n", g_manual_target,
             (g_manual_target == 0) ? "parado" : "mexendo");
      if (g_federated_enabled) {
        serial_dump_model();
      }
      print_command_menu();
    }
    g_window_count = 0;
    sl_sleeptimer_delay_millisecond(40U);
    return;
  }

  printf("features accel: %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
         features[0], features[1], features[2], features[3],
         features[4], features[5], features[6]);
  printf("features gyro: %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
         features[7], features[8], features[9], features[10],
         features[11], features[12], features[13]);

  printf("sample=%lu score=%.4f predicted_class=%d\n",
         (unsigned long)g_sample_count,
         score,
         predicted_label);

  if (g_mode == ML_DEMO_LEO_MODE_INFERENCE) {
    const char *class_name = (predicted_label == 0) ? "parado" : "mexendo";
    printf("mode=inference => class=%d (%s), no weight update\n", predicted_label, class_name);
  } else if (g_mode == ML_DEMO_LEO_MODE_LOCAL_TRAINING) {
    const char *class_name = (g_manual_target == 0) ? "parado" : "mexendo";
    if (training_window) {
      printf("mode=local_training model=%s federated=%s => training class=%d (%s), remaining_windows=%lu\n",
             (g_model_type == ML_DEMO_LEO_MODEL_CNN) ? "cnn" : "linear",
             g_federated_enabled ? "on" : "off",
             g_manual_target,
             class_name,
             (unsigned long)g_training_windows_remaining);
    } else {
      printf("mode=local_training model=%s federated=%s => waiting for next command\n",
             (g_model_type == ML_DEMO_LEO_MODEL_CNN) ? "cnn" : "linear",
             g_federated_enabled ? "on" : "off");
    }
  }

  if (training_window && g_training_windows_remaining == 0U) {
    printf("training complete: class=%d (%s)\n", g_manual_target,
           (g_manual_target == 0) ? "parado" : "mexendo");
    if (g_federated_enabled) {
      serial_dump_model();
    }
    print_command_menu();
  }

  g_window_count = 0;
  sl_sleeptimer_delay_millisecond(40U);
}
