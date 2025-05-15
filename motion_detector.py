# -*- coding: utf-8 -*-
import time
import datetime
import cv2 as cv
import os
import argparse # コマンドライン引数処理のため
from discord_webhook import DiscordWebhook

# --- 設定項目 ---
SAVE_DIR = './image/'
FN_SUFFIX = 'motion_detect.jpg'
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DELTA_MAX = 255
DOT_TH = 20
MOTION_FACTOR_TH = 0.05
ACCUMULATE_WEIGHT = 0.5
KEY_SET_BACKGROUND = ord('s') # GUIモード時のみ有効
KEY_QUIT = 27                 # GUIモード時のみ有効
HEADLESS_LOOP_DELAY = 0.1     # ヘッドレスモード時のループ遅延（秒）
# --- ここまで設定項目 ---

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def main():
    global avg_background
    avg_background = None
    motion_detection_enabled = False

    # --- コマンドライン引数の解析 ---
    parser = argparse.ArgumentParser(description="OpenCV Motion Detector")
    parser.add_argument(
        '--gui',
        action='store_true',  # 指定されると display_gui が True になる
        dest='display_gui',
        help="Enable GUI display"
    )
    parser.set_defaults(display_gui=False) # デフォルトはGUI表示OFF (ヘッドレスモード)
    args = parser.parse_args()
    # --- ここまでコマンドライン引数の解析 ---

    ensure_dir(SAVE_DIR)

    cap = cv.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Starting motion detection module.")
    if args.display_gui:
        print(f"GUI display enabled. Press '{chr(KEY_SET_BACKGROUND)}' in the window to set/reset background.")
        print(f"Press 'Esc' in the window to quit.")
    else:
        print("GUI display disabled (headless mode default).")
        print("Setting initial background automatically...")
        ret, frame_for_bg = cap.read()
        if ret:
            gray_for_bg = cv.cvtColor(frame_for_bg, cv.COLOR_BGR2GRAY)
            avg_background = gray_for_bg.copy().astype("float")
            motion_detection_enabled = True
            print(f"Initial background set. Motion detection enabled.")
        else:
            print("Error: Could not read frame for initial background in headless mode. Exiting.")
            cap.release()
            return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            dt_now = datetime.datetime.now()
            dt_format_string = dt_now.strftime('%Y-%m-%d %H:%M:%S')
            
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # GUIモードで背景未設定の場合のメッセージ (実際にはこの分岐には入りにくい)
            if avg_background is None and args.display_gui:
                cv.putText(frame, f"Press '{chr(KEY_SET_BACKGROUND)}' to set background", (25, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif avg_background is not None: # 背景が設定されていれば検知処理
                cv.accumulateWeighted(gray, avg_background, ACCUMULATE_WEIGHT)
                frame_delta = cv.absdiff(gray, cv.convertScaleAbs(avg_background))
                thresh = cv.threshold(frame_delta, DOT_TH, DELTA_MAX, cv.THRESH_BINARY)[1]
                motion_factor = thresh.sum() / (thresh.size * DELTA_MAX)
                motion_factor_str = f'{motion_factor:.08f}'
                
                if args.display_gui: # GUIが有効な場合のみフレームにモーションファクターを描画
                    cv.putText(frame, motion_factor_str, (25, FRAME_HEIGHT - 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if motion_detection_enabled and motion_factor > MOTION_FACTOR_TH:
                    f_name = dt_now.strftime('%Y%m%d%H%M%S%f') + "_" + FN_SUFFIX
                    save_path = os.path.join(SAVE_DIR, f_name)
                    
                    # 保存するフレームには日付を描画 (GUIの有無に関わらず)
                    frame_to_save = frame.copy()
                    cv.putText(frame_to_save, dt_format_string, (25, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv.imwrite(save_path, frame_to_save)
                    print(f"DETECTED: {f_name} (Factor: {motion_factor_str})")
                
                if args.display_gui: # GUIが有効な場合のみ輪郭を描画
                    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

            if args.display_gui:
                cv.imshow('Motion Detection Camera', frame)
                key = cv.waitKey(1) & 0xFF # GUIモードでは短い待機時間でキー入力をチェック
                
                if key == KEY_QUIT:
                    print("Exit key pressed.")
                    break
                elif key == KEY_SET_BACKGROUND:
                    avg_background = gray.copy().astype("float")
                    motion_detection_enabled = True # 再度有効化
                    print(f"Background re-set at {dt_format_string}. Motion detection enabled.")
            else:
                # ヘッドレスモード時は、ループの最後に短いスリープを入れる
                time.sleep(HEADLESS_LOOP_DELAY)

    except KeyboardInterrupt: # Ctrl+C で終了できるようにする
        print("Interrupted by user (Ctrl+C). Exiting...")
    finally:
        print("Exiting motion detection module.")
        cap.release()
        if args.display_gui:
            cv.destroyAllWindows()

if __name__ == '__main__':
    main()