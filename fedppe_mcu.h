#ifndef FEDPPE_MCU_H
#define FEDPPE_MCU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define FEDPPE_BLE_STATUS_SIZE 42U
#define FEDPPE_BLE_HEAD_SIZE  112U

void fedppe_mcu_init(void);
void fedppe_mcu_process(void);

bool fedppe_mcu_ble_set_head(const uint8_t *payload, size_t length);
bool fedppe_mcu_ble_start_training(uint8_t label,
                                   uint32_t windows,
                                   bool reset_sample_count);
void fedppe_mcu_ble_get_status(uint8_t payload[FEDPPE_BLE_STATUS_SIZE]);
void fedppe_mcu_ble_get_head(uint8_t payload[FEDPPE_BLE_HEAD_SIZE]);

#endif
