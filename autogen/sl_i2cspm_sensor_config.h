/***************************************************************************//**
 * @file
 * @brief I2CSPM configuration for the BRD2601B Hall sensor.
 ******************************************************************************/

#ifndef SL_I2CSPM_SENSOR_CONFIG_H
#define SL_I2CSPM_SENSOR_CONFIG_H

#define SL_I2CSPM_SENSOR_REFERENCE_CLOCK 0
#define SL_I2CSPM_SENSOR_SPEED_MODE      0

#define SL_I2CSPM_SENSOR_PERIPHERAL      I2C1
#define SL_I2CSPM_SENSOR_PERIPHERAL_NO   1
#define SL_I2CSPM_SENSOR_SCL_PORT        SL_GPIO_PORT_C
#define SL_I2CSPM_SENSOR_SCL_PIN         4
#define SL_I2CSPM_SENSOR_SDA_PORT        SL_GPIO_PORT_C
#define SL_I2CSPM_SENSOR_SDA_PIN         5

#endif
