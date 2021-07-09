/*
  update mavlink parser
 */


#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))


int mav_update(int dev_fd);

void mav_set_message_rate(uint32_t message_id, float rate_hz);
void send_heartbeat(void);

bool get_free_msg_buf_index(uint8_t &index);





enum class TypeMask: uint8_t {
    VISION_POSITION_ESTIMATE   = (1 << 0),
    VISION_SPEED_ESTIMATE      = (1 << 1),
    VISION_POSITION_DELTA      = (1 << 2)
};

bool should_send(TypeMask type_mask);
void update_vp_estimate();
// void update_vp_estimate(const Location &loc,
//                         const Vector3f &position,
//                         const Vector3f &velocity,
//                         const Quaternion &attitude);

void maybe_send_heartbeat();


