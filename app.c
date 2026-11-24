/***************************************************************************//**
 * @file
 * @brief Top level application functions
 *******************************************************************************
 * # License
 * <b>Copyright 2022 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc. Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement. This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/
#include "app.h"
#include "fedppe_mcu.h"
#include "gatt_db.h"
#include "sl_bt_api.h"

#include <stdbool.h>
#include <stdio.h>

#define BLE_MAX_ATT_VALUE      20U
#define BLE_FRAME_HEADER_SIZE   5U
#define BLE_FRAME_DATA_SIZE    (BLE_MAX_ATT_VALUE - BLE_FRAME_HEADER_SIZE)
#define BLE_PROTOCOL_MAGIC    0xa5U
#define BLE_MAX_MESSAGE_SIZE  128U

#define BLE_OP_STATUS       1U
#define BLE_OP_SET_HEAD     2U
#define BLE_OP_TRAIN        3U
#define BLE_OP_GET_HEAD     4U
#define BLE_RESPONSE_BIT  0x80U

static uint8_t g_advertising_set = 0xffU;
static uint8_t g_ble_connection;
static bool g_ble_connected;
static bool g_ble_notifications_enabled;
static uint8_t g_rx_opcode;
static uint8_t g_rx_transaction;
static uint8_t g_rx_total;
static uint8_t g_rx_received;
static uint8_t g_rx_message[BLE_MAX_MESSAGE_SIZE];

static void start_ble_advertising(void)
{
  sl_status_t status;

  status = sl_bt_advertiser_create_set(&g_advertising_set);
  if (status != SL_STATUS_OK) {
    printf("BLE_ERROR advertiser create status=%lu\n", (unsigned long)status);
    return;
  }

  status = sl_bt_legacy_advertiser_generate_data(
    g_advertising_set,
    sl_bt_advertiser_general_discoverable);
  if (status != SL_STATUS_OK) {
    printf("BLE_ERROR advertiser data status=%lu\n", (unsigned long)status);
    return;
  }

  status = sl_bt_advertiser_set_timing(g_advertising_set, 160U, 160U, 0U, 0U);
  if (status != SL_STATUS_OK) {
    printf("BLE_ERROR advertiser timing status=%lu\n", (unsigned long)status);
    return;
  }

  status = sl_bt_legacy_advertiser_start(
    g_advertising_set,
    sl_bt_legacy_advertiser_connectable);
  if (status != SL_STATUS_OK) {
    printf("BLE_ERROR advertiser start status=%lu\n", (unsigned long)status);
    return;
  }

  printf("BLE_READY name=TCC001-BLE\n");
}

static void send_ble_response(uint8_t opcode,
                              uint8_t transaction,
                              const uint8_t *payload,
                              uint8_t payload_length)
{
  uint8_t offset = 0U;

  if (!g_ble_connected || !g_ble_notifications_enabled) {
    printf("BLE_ERROR notifications are not enabled\n");
    return;
  }

  do {
    uint8_t frame[BLE_MAX_ATT_VALUE];
    uint8_t data_length = (uint8_t)(payload_length - offset);
    if (data_length > BLE_FRAME_DATA_SIZE) {
      data_length = BLE_FRAME_DATA_SIZE;
    }

    frame[0] = BLE_PROTOCOL_MAGIC;
    frame[1] = (uint8_t)(opcode | BLE_RESPONSE_BIT);
    frame[2] = transaction;
    frame[3] = offset;
    frame[4] = payload_length;
    if (data_length > 0U) {
      memcpy(&frame[BLE_FRAME_HEADER_SIZE], &payload[offset], data_length);
    }

    sl_status_t status = sl_bt_gatt_server_write_attribute_value(
      gattdb_tcc_ble_tx,
      0U,
      BLE_FRAME_HEADER_SIZE + data_length,
      frame);
    if (status == SL_STATUS_OK) {
      status = sl_bt_gatt_server_send_notification(
        g_ble_connection,
        gattdb_tcc_ble_tx,
        BLE_FRAME_HEADER_SIZE + data_length,
        frame);
    }
    if (status != SL_STATUS_OK) {
      printf("BLE_ERROR response notification status=%lu\n",
             (unsigned long)status);
      return;
    }
    offset = (uint8_t)(offset + data_length);
  } while (offset < payload_length);
}

static uint32_t read_u32_le(const uint8_t *data)
{
  return (uint32_t)data[0]
         | ((uint32_t)data[1] << 8)
         | ((uint32_t)data[2] << 16)
         | ((uint32_t)data[3] << 24);
}

static void handle_ble_request(uint8_t opcode,
                               uint8_t transaction,
                               const uint8_t *payload,
                               uint8_t payload_length)
{
  uint8_t response[FEDPPE_BLE_HEAD_SIZE + 1U];
  uint8_t response_length = 1U;
  response[0] = 0U;

  switch (opcode) {
    case BLE_OP_STATUS:
      if (payload_length != 0U) {
        response[0] = 1U;
        break;
      }
      fedppe_mcu_ble_get_status(&response[1]);
      response_length += FEDPPE_BLE_STATUS_SIZE;
      break;

    case BLE_OP_SET_HEAD:
      if (!fedppe_mcu_ble_set_head(payload, payload_length)) {
        response[0] = 1U;
      }
      break;

    case BLE_OP_TRAIN:
      if (payload_length != 6U
          || payload[5] > 1U
          || !fedppe_mcu_ble_start_training(payload[0],
                                             read_u32_le(&payload[1]),
                                             payload[5] != 0U)) {
        response[0] = 1U;
      }
      break;

    case BLE_OP_GET_HEAD:
      if (payload_length != 0U) {
        response[0] = 1U;
        break;
      }
      fedppe_mcu_ble_get_head(&response[1]);
      response_length += FEDPPE_BLE_HEAD_SIZE;
      break;

    default:
      response[0] = 1U;
      break;
  }

  send_ble_response(opcode, transaction, response, response_length);
}

static void process_ble_protocol_frame(const uint8_t *frame, size_t frame_length)
{
  uint8_t opcode;
  uint8_t transaction;
  uint8_t offset;
  uint8_t total;
  uint8_t data_length;

  if (frame_length < BLE_FRAME_HEADER_SIZE
      || frame[0] != BLE_PROTOCOL_MAGIC) {
    printf("BLE_ERROR malformed protocol frame\n");
    return;
  }

  opcode = frame[1];
  transaction = frame[2];
  offset = frame[3];
  total = frame[4];
  data_length = (uint8_t)(frame_length - BLE_FRAME_HEADER_SIZE);
  if (total > BLE_MAX_MESSAGE_SIZE
      || data_length > BLE_FRAME_DATA_SIZE
      || (uint16_t)offset + data_length > total) {
    printf("BLE_ERROR invalid protocol frame length\n");
    return;
  }

  if (offset == 0U) {
    g_rx_opcode = opcode;
    g_rx_transaction = transaction;
    g_rx_total = total;
    g_rx_received = 0U;
  }
  if (opcode != g_rx_opcode
      || transaction != g_rx_transaction
      || total != g_rx_total
      || offset != g_rx_received) {
    printf("BLE_ERROR unexpected protocol fragment\n");
    g_rx_received = 0U;
    return;
  }

  if (data_length > 0U) {
    memcpy(&g_rx_message[offset], &frame[BLE_FRAME_HEADER_SIZE], data_length);
  }
  g_rx_received = (uint8_t)(g_rx_received + data_length);
  if (g_rx_received == g_rx_total) {
    handle_ble_request(g_rx_opcode,
                       g_rx_transaction,
                       g_rx_message,
                       g_rx_total);
    g_rx_received = 0U;
  }
}

static void handle_ble_write(void)
{
  uint8_t payload[BLE_MAX_ATT_VALUE];
  size_t payload_length = 0U;
  sl_status_t status;

  status = sl_bt_gatt_server_read_attribute_value(gattdb_tcc_ble_rx,
                                                   0U,
                                                   sizeof(payload),
                                                   &payload_length,
                                                   payload);
  if (status != SL_STATUS_OK) {
    printf("BLE_ERROR read status=%lu\n", (unsigned long)status);
    return;
  }

  if (payload_length > 0U && payload[0] == BLE_PROTOCOL_MAGIC) {
    process_ble_protocol_frame(payload, payload_length);
    return;
  }

  status = sl_bt_gatt_server_write_attribute_value(gattdb_tcc_ble_tx,
                                                    0U,
                                                    payload_length,
                                                    payload);
  if (status != SL_STATUS_OK) {
    printf("BLE_ERROR response write status=%lu\n", (unsigned long)status);
    return;
  }

  if (g_ble_connected && g_ble_notifications_enabled) {
    status = sl_bt_gatt_server_send_notification(g_ble_connection,
                                                 gattdb_tcc_ble_tx,
                                                 payload_length,
                                                 payload);
    if (status != SL_STATUS_OK) {
      printf("BLE_ERROR notification status=%lu\n", (unsigned long)status);
      return;
    }
  }

  printf("BLE_ECHO bytes=%lu\n", (unsigned long)payload_length);
}

/***************************************************************************//**
 * Initialize application.
 ******************************************************************************/
void app_init(void)
{
  fedppe_mcu_init();
}

/***************************************************************************//**
 * App ticking function.
 ******************************************************************************/
void app_process_action(void)
{
  fedppe_mcu_process();
}

void sl_bt_on_event(sl_bt_msg_t *event)
{
  sl_status_t status;

  switch (SL_BT_MSG_ID(event->header)) {
    case sl_bt_evt_system_boot_id:
      start_ble_advertising();
      break;
    
    case sl_bt_evt_connection_opened_id:
      g_ble_connection = event->data.evt_connection_opened.connection;
      g_ble_connected = true;
      g_ble_notifications_enabled = false;
      g_rx_received = 0U;
      printf("BLE_CONNECTED\n");
      break;

    case sl_bt_evt_connection_closed_id:
      g_ble_connected = false;
      g_ble_notifications_enabled = false;
      g_rx_received = 0U;
      printf("BLE_DISCONNECTED\n");
      status = sl_bt_legacy_advertiser_generate_data(
        g_advertising_set,
        sl_bt_advertiser_general_discoverable);
      if (status == SL_STATUS_OK) {
        status = sl_bt_legacy_advertiser_start(
          g_advertising_set,
          sl_bt_legacy_advertiser_connectable);
      }
      if (status != SL_STATUS_OK) {
        printf("BLE_ERROR restart advertising status=%lu\n",
               (unsigned long)status);
      }
      break;

    case sl_bt_evt_gatt_server_characteristic_status_id:
      if (event->data.evt_gatt_server_characteristic_status.characteristic
          == gattdb_tcc_ble_tx) {
        g_ble_notifications_enabled =
          (event->data.evt_gatt_server_characteristic_status.client_config_flags
           & sl_bt_gatt_notification) != 0U;
      }
      break;

    case sl_bt_evt_gatt_server_attribute_value_id:
      if (event->data.evt_gatt_server_attribute_value.attribute
          == gattdb_tcc_ble_rx) {
        handle_ble_write();
      }
      break;

    default:
      break;
  }
}
