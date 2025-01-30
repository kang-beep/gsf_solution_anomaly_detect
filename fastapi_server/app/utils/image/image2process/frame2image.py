import cv2

def save_frame_at_time(video_path, output_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(fps * time_sec)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Frame saved to {output_path}")
    else:
        print("Failed to extract frame")
    
    cap.release()

video_path = "/home/gsf/Desktop/kangsan_workspace/gsf_solution/fastapi_server/app/utils/image/wobble/wobble_0.1mm.mp4"
output_path = "/home/gsf/Desktop/kangsan_workspace/gsf_solution/fastapi_server/app/utils/image/image2process/target.png"
save_frame_at_time(video_path, output_path, 50)