#ifndef ML_DEMO_LEO_H
#define ML_DEMO_LEO_H

#include <stdbool.h>
#include <stdint.h>

#define ML_DEMO_LEO_FEATURES_PER_SENSOR 7
#define ML_DEMO_LEO_FEATURE_COUNT (2 * ML_DEMO_LEO_FEATURES_PER_SENSOR)
#define ML_DEMO_LEO_WINDOW_SIZE   32
#define ML_DEMO_LEO_CNN_FILTERS   2
#define ML_DEMO_LEO_CNN_KERNEL    3
#define ML_DEMO_LEO_CNN_CHANNELS  6

typedef enum {
  ML_DEMO_LEO_MODE_INFERENCE = 0,
  ML_DEMO_LEO_MODE_LOCAL_TRAINING = 1,
} ml_demo_leo_mode_t;

typedef enum {
  ML_DEMO_LEO_MODEL_LINEAR = 0,
  ML_DEMO_LEO_MODEL_CNN = 1,
} ml_demo_leo_model_type_t;

typedef struct {
  float weights[ML_DEMO_LEO_FEATURE_COUNT];
  float bias;
  float cnn_weights[ML_DEMO_LEO_CNN_FILTERS][ML_DEMO_LEO_CNN_KERNEL][ML_DEMO_LEO_CNN_CHANNELS];
  float cnn_bias[ML_DEMO_LEO_CNN_FILTERS];
  float cnn_output_weights[ML_DEMO_LEO_CNN_FILTERS];
  float cnn_output_bias;
} ml_demo_leo_linear_model_t;

void ml_demo_leo_init(void);
void ml_demo_leo_set_mode(ml_demo_leo_mode_t mode);
void ml_demo_leo_set_manual_target(int target);
void ml_demo_leo_set_model(ml_demo_leo_model_type_t model_type);
void ml_demo_leo_set_federated(bool enabled);
void ml_demo_leo_process(void);

#endif // ML_DEMO_LEO_H
