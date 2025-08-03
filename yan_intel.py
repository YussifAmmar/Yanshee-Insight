import chat
import YanAPI
import time

YanAPI.yan_api_init("192.168.1.26")
while True:
    print(YanAPI.get_robot_volume())

    response = chat.action_on_chat()

    YanAPI.start_voice_tts(tts = response.resp, interrupt=False)

    for i in range(len(response.move_name)):
        res = YanAPI.start_play_motion(   name = response.move_name[i],
                                    direction = response.move_direction[i],
                                    repeat= response.move_repeat[i],
                                    speed= response.move_speed[i],
                                )
        time.sleep(res['data']["total_time"] / 1000)
        YanAPI.start_play_motion()
        print("out")
