#include "fedppe_mcu.h"

#include "accelerometer.h"
#include "sl_iostream.h"
#include "sl_sleeptimer.h"
#include "sl_status.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FEDPPE_WINDOW_SIZE       32U
#define FEDPPE_FEATURES          14U
#define FEDPPE_HIDDEN_1          16U
#define FEDPPE_EMBEDDING_SIZE     8U
#define FEDPPE_CLASSES            3U
#define FEDPPE_BACKBONE_PARAMS  404U
#define FEDPPE_HEAD_PARAMS       27U
#define FEDPPE_COMMAND_SIZE      80U
#define FEDPPE_LEARNING_RATE   0.02f

#include "fedppe_backbone.h"

typedef struct {
  float weights[FEDPPE_CLASSES][FEDPPE_EMBEDDING_SIZE];
  float bias[FEDPPE_CLASSES];
} fedppe_head_t;

static imu_data_t g_window[FEDPPE_WINDOW_SIZE];
static uint32_t g_window_count;
static uint32_t g_training_windows;
static uint32_t g_training_samples;
static int g_training_label = -1;
static char g_command[FEDPPE_COMMAND_SIZE];
static size_t g_command_length;
static bool g_sensor_ready;
static fedppe_head_t g_head;

static const char *const g_class_names[FEDPPE_CLASSES] = {
  "parado", "carregando", "vestindo"
};

static float channel_value(const imu_data_t *sample, uint32_t channel)
{
  switch (channel) {
    case 0U: return sample->x;
    case 1U: return sample->y;
    case 2U: return sample->z;
    case 3U: return sample->gx;
    case 4U: return sample->gy;
    default: return sample->gz;
  }
}

static void compute_sensor_features(const imu_data_t *window,
                                    uint32_t first_channel,
                                    float output[7])
{
  float sum_magnitude = 0.0f;
  float sum_magnitude_squared = 0.0f;
  float sum_delta = 0.0f;
  float sum_delta_squared = 0.0f;
  float sum_axis[3] = { 0.0f, 0.0f, 0.0f };
  float sum_axis_squared[3] = { 0.0f, 0.0f, 0.0f };

  for (uint32_t i = 0U; i < FEDPPE_WINDOW_SIZE; ++i) {
    float axis[3] = {
      channel_value(&window[i], first_channel),
      channel_value(&window[i], first_channel + 1U),
      channel_value(&window[i], first_channel + 2U)
    };
    float magnitude = sqrtf(axis[0] * axis[0]
                            + axis[1] * axis[1]
                            + axis[2] * axis[2]);
    sum_magnitude += magnitude;
    sum_magnitude_squared += magnitude * magnitude;

    for (uint32_t j = 0U; j < 3U; ++j) {
      sum_axis[j] += axis[j];
      sum_axis_squared[j] += axis[j] * axis[j];
    }

    if (i > 0U) {
      float dx = axis[0] - channel_value(&window[i - 1U], first_channel);
      float dy = axis[1] - channel_value(&window[i - 1U], first_channel + 1U);
      float dz = axis[2] - channel_value(&window[i - 1U], first_channel + 2U);
      float delta = sqrtf(dx * dx + dy * dy + dz * dz);
      sum_delta += delta;
      sum_delta_squared += delta * delta;
    }
  }

  const float inv_count = 1.0f / (float)FEDPPE_WINDOW_SIZE;
  const float inv_delta_count = 1.0f / (float)(FEDPPE_WINDOW_SIZE - 1U);
  const float mean_magnitude = sum_magnitude * inv_count;
  const float mean_delta = sum_delta * inv_delta_count;

  output[0] = mean_magnitude * 0.001f;
  output[1] = sqrtf(fmaxf(0.0f,
                           sum_magnitude_squared * inv_count
                           - mean_magnitude * mean_magnitude)) * 0.001f;
  output[2] = mean_delta * 0.001f;
  output[3] = sqrtf(fmaxf(0.0f,
                           sum_delta_squared * inv_delta_count
                           - mean_delta * mean_delta)) * 0.001f;

  for (uint32_t j = 0U; j < 3U; ++j) {
    float mean_axis = sum_axis[j] * inv_count;
    output[4U + j] = sqrtf(fmaxf(0.0f,
                                  sum_axis_squared[j] * inv_count
                                  - mean_axis * mean_axis)) * 0.001f;
  }
}

static void compute_features(const imu_data_t *window,
                             float output[FEDPPE_FEATURES])
{
  compute_sensor_features(window, 0U, output);
  compute_sensor_features(window, 3U, &output[7]);
}

static float relu(float value)
{
  return (value > 0.0f) ? value : 0.0f;
}

static void backbone_forward(const float features[FEDPPE_FEATURES],
                             float embedding[FEDPPE_EMBEDDING_SIZE])
{
  float hidden[FEDPPE_HIDDEN_1];

  for (uint32_t i = 0U; i < FEDPPE_HIDDEN_1; ++i) {
    float value = g_b1[i];
    for (uint32_t j = 0U; j < FEDPPE_FEATURES; ++j) {
      float normalized = (features[j] - g_input_mean[j])
                         * g_input_inverse_std[j];
      value += g_w1[i][j] * normalized;
    }
    hidden[i] = relu(value);
  }

  for (uint32_t i = 0U; i < FEDPPE_EMBEDDING_SIZE; ++i) {
    float value = g_b2[i];
    for (uint32_t j = 0U; j < FEDPPE_HIDDEN_1; ++j) {
      value += g_w2[i][j] * hidden[j];
    }
    embedding[i] = relu(value);
  }
}

static void head_predict(const float embedding[FEDPPE_EMBEDDING_SIZE],
                         float probabilities[FEDPPE_CLASSES])
{
  float max_logit = -INFINITY;
  float sum = 0.0f;

  for (uint32_t c = 0U; c < FEDPPE_CLASSES; ++c) {
    float logit = g_head.bias[c];
    for (uint32_t j = 0U; j < FEDPPE_EMBEDDING_SIZE; ++j) {
      logit += g_head.weights[c][j] * embedding[j];
    }
    probabilities[c] = logit;
    if (logit > max_logit) {
      max_logit = logit;
    }
  }

  for (uint32_t c = 0U; c < FEDPPE_CLASSES; ++c) {
    probabilities[c] = expf(probabilities[c] - max_logit);
    sum += probabilities[c];
  }
  if (sum <= 0.0f) {
    printf("FEDPPE_ERROR softmax normalization failed\n");
    return;
  }
  for (uint32_t c = 0U; c < FEDPPE_CLASSES; ++c) {
    probabilities[c] /= sum;
  }
}

static void train_head(const float embedding[FEDPPE_EMBEDDING_SIZE],
                       uint32_t label)
{
  float probabilities[FEDPPE_CLASSES];
  float errors[FEDPPE_CLASSES];
  head_predict(embedding, probabilities);

  for (uint32_t c = 0U; c < FEDPPE_CLASSES; ++c) {
    errors[c] = probabilities[c] - ((c == label) ? 1.0f : 0.0f);
  }

  for (uint32_t c = 0U; c < FEDPPE_CLASSES; ++c) {
    for (uint32_t j = 0U; j < FEDPPE_EMBEDDING_SIZE; ++j) {
      g_head.weights[c][j] -= FEDPPE_LEARNING_RATE * errors[c] * embedding[j];
    }
    g_head.bias[c] -= FEDPPE_LEARNING_RATE * errors[c];
  }
}

static bool set_head_parameter(uint32_t index, float value)
{
  const uint32_t weights_count = FEDPPE_CLASSES * FEDPPE_EMBEDDING_SIZE;
  if (index >= FEDPPE_HEAD_PARAMS || !isfinite(value)) {
    return false;
  }
  if (index < weights_count) {
    g_head.weights[index / FEDPPE_EMBEDDING_SIZE]
                   [index % FEDPPE_EMBEDDING_SIZE] = value;
  } else {
    g_head.bias[index - weights_count] = value;
  }
  return true;
}

static float get_head_parameter(uint32_t index)
{
  const uint32_t weights_count = FEDPPE_CLASSES * FEDPPE_EMBEDDING_SIZE;
  if (index < weights_count) {
    return g_head.weights[index / FEDPPE_EMBEDDING_SIZE]
                          [index % FEDPPE_EMBEDDING_SIZE];
  }
  return g_head.bias[index - weights_count];
}

bool fedppe_mcu_ble_set_head(const uint8_t *payload, size_t length)
{
  float parameters[FEDPPE_HEAD_PARAMS];

  if (payload == NULL || length != sizeof(parameters)) {
    return false;
  }

  for (uint32_t i = 0U; i < FEDPPE_HEAD_PARAMS; ++i) {
    memcpy(&parameters[i], &payload[i * sizeof(float)], sizeof(float));
    if (!isfinite(parameters[i])) {
      return false;
    }
  }

  for (uint32_t i = 0U; i < FEDPPE_HEAD_PARAMS; ++i) {
    if (!set_head_parameter(i, parameters[i])) {
      return false;
    }
  }
  return true;
}

bool fedppe_mcu_ble_start_training(uint8_t label,
                                   uint32_t windows,
                                   bool reset_sample_count)
{
  if (!g_sensor_ready || label >= FEDPPE_CLASSES || windows == 0U
      || g_training_windows != 0U) {
    return false;
  }

  g_training_label = (int)label;
  g_training_windows = windows;
  if (reset_sample_count) {
    g_training_samples = 0U;
  }
  return true;
}

void fedppe_mcu_ble_get_status(uint8_t payload[FEDPPE_BLE_STATUS_SIZE])
{
  const uint16_t backbone_params = FEDPPE_BACKBONE_PARAMS;
  const uint16_t head_params = FEDPPE_HEAD_PARAMS;
  const uint32_t samples = g_training_samples;

  payload[0] = 1U;
  memcpy(&payload[1], &backbone_params, sizeof(backbone_params));
  memcpy(&payload[3], &head_params, sizeof(head_params));
  memcpy(&payload[5], &samples, sizeof(samples));
  payload[9] = (g_training_windows > 0U) ? 1U : 0U;
  memcpy(&payload[10], FEDPPE_BACKBONE_FINGERPRINT, 32U);
}

void fedppe_mcu_ble_get_head(uint8_t payload[FEDPPE_BLE_HEAD_SIZE])
{
  const uint32_t samples = g_training_samples;

  memcpy(payload, &samples, sizeof(samples));
  for (uint32_t i = 0U; i < FEDPPE_HEAD_PARAMS; ++i) {
    const float value = get_head_parameter(i);
    memcpy(&payload[4U + i * sizeof(float)], &value, sizeof(value));
  }
}

static void dump_head(void)
{
  printf("FL_MODEL_BEGIN %u %lu\n",
         FEDPPE_HEAD_PARAMS,
         (unsigned long)g_training_samples);
  for (uint32_t i = 0U; i < FEDPPE_HEAD_PARAMS; ++i) {
    printf("FL_WEIGHT %lu %.9g\n", (unsigned long)i, get_head_parameter(i));
  }
  printf("FL_MODEL_END\n");
}

static void execute_command(void)
{
  unsigned long index;
  unsigned long count;
  float value;
  int label;
  char trailing;

  g_command[g_command_length] = '\0';

  if (strcmp(g_command, "status") == 0) {
    printf("FEDPPE_STATUS backbone=%u/%u fingerprint=%s head=%u samples=%lu\n",
           FEDPPE_BACKBONE_PARAMS,
           FEDPPE_BACKBONE_PARAMS,
           FEDPPE_BACKBONE_FINGERPRINT,
           FEDPPE_HEAD_PARAMS,
           (unsigned long)g_training_samples);
  } else if (sscanf(g_command, "H %lu %f %c", &index, &value, &trailing) == 2) {
    if (index < FEDPPE_HEAD_PARAMS
        && set_head_parameter((uint32_t)index, value)) {
      printf("H_ACK %lu\n", index);
    } else {
      printf("FEDPPE_ERROR invalid head parameter\n");
    }
  } else if (strcmp(g_command, "fl_dump") == 0) {
    dump_head();
  } else if (sscanf(g_command, "train %d %lu %c", &label, &count, &trailing) == 2) {
    if (label < 0 || label >= (int)FEDPPE_CLASSES
        || count == 0UL || count > UINT32_MAX) {
      printf("FEDPPE_ERROR use train CLASS WINDOWS, CLASS=0..2\n");
    } else if (!fedppe_mcu_ble_start_training((uint8_t)label,
                                               (uint32_t)count,
                                               true)) {
      printf("FEDPPE_ERROR IMU not ready or training already active\n");
    } else {
      printf("FEDPPE_TRAINING class=%d name=%s windows=%lu\n",
             label, g_class_names[label], count);
    }
  } else if (strcmp(g_command, "help") == 0) {
    printf("commands: status, H INDEX VALUE, train CLASS WINDOWS, fl_dump\n");
    printf("classes: 0=parado 1=carregando 2=vestindo\n");
  } else if (g_command_length != 0U) {
    printf("FEDPPE_ERROR unknown command\n");
  }

  g_command_length = 0U;
}

static void poll_serial(void)
{
  char received;

  while (sl_iostream_getchar(SL_IOSTREAM_STDIN, &received) == SL_STATUS_OK) {
    if (received == '\r' || received == '\n') {
      execute_command();
    } else if (g_command_length < sizeof(g_command) - 1U) {
      g_command[g_command_length++] = received;
    } else {
      g_command_length = 0U;
      printf("FEDPPE_ERROR command too long\n");
    }
  }
}

void fedppe_mcu_init(void)
{
  sl_status_t status = accelerometer_setup();
  if (status != SL_STATUS_OK) {
    printf("FEDPPE_ERROR accelerometer setup failed status=%lu\n",
           (unsigned long)status);
    return;
  }

  memset(&g_head, 0, sizeof(g_head));
  g_window_count = 0U;
  g_training_windows = 0U;
  g_training_samples = 0U;
  g_training_label = -1;
  g_sensor_ready = true;
  printf("FEDPPE_READY window=%u features=%u hidden=%u embedding=%u classes=%u\n",
         FEDPPE_WINDOW_SIZE,
         FEDPPE_FEATURES,
         FEDPPE_HIDDEN_1,
         FEDPPE_EMBEDDING_SIZE,
         FEDPPE_CLASSES);
  printf("FEDPPE_INFO labels 0=parado 1=carregando 2=vestindo; manual labels for prototype\n");
}

void fedppe_mcu_process(void)
{
  imu_data_t sample;
  sl_status_t status;

  poll_serial();
  if (!g_sensor_ready) {
    return;
  }

  status = imu_read(&sample, 1);
  if (status != SL_STATUS_OK) {
    return;
  }

  g_window[g_window_count++] = sample;
  if (g_window_count < FEDPPE_WINDOW_SIZE) {
    sl_sleeptimer_delay_millisecond(40U);
    return;
  }
  g_window_count = 0U;

  float features[FEDPPE_FEATURES];
  float embedding[FEDPPE_EMBEDDING_SIZE];
  float probabilities[FEDPPE_CLASSES];
  uint32_t predicted = 0U;

  compute_features(g_window, features);
  backbone_forward(features, embedding);

  if (g_training_windows > 0U && g_training_label >= 0) {
    train_head(embedding, (uint32_t)g_training_label);
    --g_training_windows;
    ++g_training_samples;
    if (g_training_windows == 0U) {
      printf("FL_LOCAL_DONE %lu\n", (unsigned long)g_training_samples);
      g_training_label = -1;
    }
  } else {
    head_predict(embedding, probabilities);
    for (uint32_t c = 1U; c < FEDPPE_CLASSES; ++c) {
      if (probabilities[c] > probabilities[predicted]) {
        predicted = c;
      }
    }
    printf("FEDPPE_PRED class=%lu name=%s p0=%.4f p1=%.4f p2=%.4f\n",
           (unsigned long)predicted,
           g_class_names[predicted],
           probabilities[0],
           probabilities[1],
           probabilities[2]);
  }

  sl_sleeptimer_delay_millisecond(40U);
}
