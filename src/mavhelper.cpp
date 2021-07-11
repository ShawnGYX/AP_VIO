
/*
  example code for mavlink from C
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <signal.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <string.h>
#include "mavhelper.h"

#define MAVLINK_USE_CONVENIENCE_FUNCTIONS

#include "../include/mavlink/mavlink_types.h"
static mavlink_system_t mavlink_system = {42,11,};

extern void comm_send_ch(mavlink_channel_t chan, uint8_t c);

#include "../include/mavlink/ardupilotmega/mavlink.h"

struct
{
    uint64_t time_send_us;
    mavlink_message_t obs_msg;
} msg_buf[3];


static mavlink_status_t mav_status;
static uint8_t target_system;

static double get_time_seconds()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return (tp.tv_sec + (tp.tv_usec*1.0e-6));
}

static double last_msg_s;
static double last_hb_s;

float gyro_factor = 1e-3;
float acc_factor = 9.81 * 1e-3;

/*
  update mavlink
 */
int mav_update(int dev_fd)
{
    uint8_t b;
    mavlink_message_t msg;

    mavlink_raw_imu_t raw_imu;
    

    while (read(dev_fd, &b, 1) == 1) {
        if (mavlink_parse_char(MAVLINK_COMM_0, b, &msg, &mav_status)) {
            double tnow = get_time_seconds();
            if (msg.msgid == MAVLINK_MSG_ID_ATTITUDE) {
                printf("msgid=%u dt=%f\n", msg.msgid, tnow - last_msg_s);
                last_msg_s = tnow;

                mavlink_msg_raw_imu_decode(&msg, &raw_imu);

                printf("xacc:%f, yacc:%f, zacc:%f, xgyro:%f, ygyro:%f, zgyro:%f.\n", raw_imu.xacc*acc_factor, raw_imu.yacc*acc_factor, raw_imu.zacc*acc_factor, raw_imu.xgyro*gyro_factor, raw_imu.ygyro*gyro_factor, raw_imu.zgyro*gyro_factor);

            }
            if (target_system == 0 && msg.msgid == MAVLINK_MSG_ID_HEARTBEAT) {
                printf("Got system ID %u\n", msg.sysid);
                target_system = msg.sysid;

                // get key messages at 200Hz
                mav_set_message_rate(MAVLINK_MSG_ID_ATTITUDE, 40);
            }
        }
    }

    double tnow = get_time_seconds();
    if (tnow - last_hb_s > 1.0) {
        last_hb_s = tnow;
        send_heartbeat();
    }

    return 0;
}

void mav_set_message_rate(uint32_t message_id, float rate_hz)
{
    mavlink_msg_command_long_send(MAVLINK_COMM_0,
                                  target_system,
                                  0,
                                  MAV_CMD_SET_MESSAGE_INTERVAL,
                                  0,
                                  message_id,
                                  (uint32_t)(1e6 / rate_hz),
                                  0, 0, 0, 0, 0);    
}

void send_heartbeat(void)
{
    printf("sending hb\n");
    mavlink_msg_heartbeat_send(
        MAVLINK_COMM_0,
        MAV_TYPE_GCS,
        MAV_AUTOPILOT_GENERIC,
        0, 0, 0);
}


// bool get_free_msg_buf_index(uint8_t &index)
// {
//     for (uint8_t i = 0; i < ARRAY_SIZE(msg_buf); i++)
//     {
//         if (msg_buf[i].time_send_us == 0)
//         {
//             index = i;
//             return true;
//         }
//     }
//     return false;
// }

// bool should_send(TypeMask type_mask)
// {
//     return 1;
// }

// void update_vp_estimate()
// // void dataStream::update_vp_estimate(const Location &loc,
// //                                     const Vector3f &position,
// //                                     const Vector3f &velocity,
// //                                     const Quaternion &attitude)
// {
//     const uint32_t now_us = get_time_seconds();

//     uint64_t last_observation_usec;
//     uint64_t time_offset_us;

//     // Calculate a random time offset to the time sent in the message
//     if (time_offset_us == 0)
//     {
//         time_offset_us = (unsigned(random()) % 7000) * 1000000ULL;
//         printf("time_off_us %llu\n", (long long unsigned)time_offset_us);
//     }

//     // send all messages in the buffer
//     bool waiting_to_send = false;
//     for (uint8_t i=0; i<ARRAY_SIZE(msg_buf); i++)
//     {
//         if ((msg_buf[i].time_send_us > 0) && (now_us >= msg_buf[i].time_send_us))
//         {
//             uint8_t buf[300];
//             uint16_t buf_len = mavlink_msg_to_send_buffer(buf, &msg_buf[i].obs_msg);
//             msg_buf[i].time_send_us = 0;

//         }
//         waiting_to_send = msg_buf[i].time_send_us != 0;

//     }
//     if (waiting_to_send) {
//         return;
//     }

//     if (now_us - last_observation_usec < 20000)
//     {
//         return;
//     }

//     // const Eigen::Quaterniond& attitude = estimatedState.pose.R().asQuaternion();
//     // const Eigen::Vector3d& position = estimatedState.pose.x();
//     // const Eigen::Vector3d& vel = estimatedState.velocity;

//     float posx = 1.0;
//     float posy = 2.0;
//     float posz = 3.0;

//     float velx = 4.0;
//     float vely = 5.0;
//     float velz = 6.0;

//     // Eigen::Vector3d euler = attitude.toRotationMatrix().eulerAngles(0,1,2);
//     // float roll = euler[0];
//     // float pitch = euler[1];
//     // float yaw = euler[2];
//     float roll = 1.0;
//     float pitch = 1.0;
//     float yaw = 1.0;
    
//     // attitude.to_euler(roll, pitch, yaw);


//     // load estimates from the filter

//     uint32_t delay_ms = 25 + unsigned(random()) % 100;
//     uint64_t time_send_us = now_us + delay_ms * 1000UL;
//     // send message
//     uint8_t msg_buf_index;
//     if (should_send(TypeMask::VISION_POSITION_ESTIMATE) && get_free_msg_buf_index(msg_buf_index))
//     {
//         mavlink_msg_vision_position_estimate_pack_chan(
//             target_system,
//             0,
//             MAVLINK_COMM_0,
//             &msg_buf[msg_buf_index].obs_msg,
//             now_us + time_offset_us,
//             posx,
//             posy,
//             posz,
//             roll,
//             pitch,
//             yaw,
//             NULL, 0);
//         msg_buf[msg_buf_index].time_send_us = time_send_us;
//     }



//     if (should_send(TypeMask::VISION_SPEED_ESTIMATE) && get_free_msg_buf_index(msg_buf_index))
//     {
//         mavlink_msg_vision_speed_estimate_pack_chan(
//             target_system,
//             0,
//             MAVLINK_COMM_0,
//             &msg_buf[msg_buf_index].obs_msg,
//             now_us + time_offset_us,
//             velx,
//             vely,
//             velz,
//             NULL, 0
//         );
//         msg_buf[msg_buf_index].time_send_us = time_send_us;
//     }

//     // uint64_t time_delta = now_us - last_observation_usec;

//     // Eigen::Quaternionf attitude_curr;

//     // attitude_curr.from_euler(roll, pitch, yaw);

//     // attitude_curr.invert();

//     // Quaternion attitude_curr_prev = attitude_curr * _attitude_prev.inverse();

//     // float angle_data[3] = {
//     //     attitude_curr_prev.get_euler_roll(),
//     //     attitude_curr_prev.get_euler_pitch(),
//     //     attitude_curr_prev.get_euler_yaw()};
    
//     // Matrix3f body_ned_m;
//     // attitude_curr.rotation_matrix(body_ned_m);

    
    

// }
