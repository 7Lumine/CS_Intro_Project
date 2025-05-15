# -*- coding: utf-8 -*-
import time
import datetime
import cv2 as cv
import os
import argparse # コマンドライン引数処理のため
# from discord_webhook import DiscordWebhook # こちらはFastAPI経由なので不要
import requests # API呼び出しに必要

#https://qiita.com/hase-k0x01/items/acd0d9159a9001ebfbd3

# --- 設定項目 ---
SAVE_DIR = './image/'
FN_SUFFIX = 'motion_detect.jpg' # 保存される画像の接尾辞 (例: motion_detect.jpg)
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
API_ENDPOINT_URL = "http://127.0.0.1:8000/send_image/" # FastAPIサーバーのエンドポイント
NOTIFICATION_COOLDOWN_SECONDS = 60 # 通知のクールダウンタイム（秒）
# --- ここまで設定項目 ---

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def send_image_to_discord_api(image_path: str, image_filename: str):
    """
    指定された画像をFastAPI経由でDiscordに送信する関数。
    """
    if not os.path.exists(image_path):
        print(f"エラー: 通知する画像ファイルが見つかりません: {image_path}")
        return False
    try:
        with open(image_path, "rb") as f:
            # APIサーバー側が 'file' というキーでファイルを受け取ることを期待
            files = {"file": (image_filename, f, "image/jpeg")} 
            response = requests.post(API_ENDPOINT_URL, files=files, timeout=10)
        
        if response.status_code == 200:
            print(f"Discordへの画像送信成功: {response.json().get('message', '成功')}")
            return True
        else:
            print(f"Discordへの画像送信失敗: ステータスコード {response.status_code}")
            try:
                print(f"エラー詳細: {response.json()}")
            except requests.exceptions.JSONDecodeError:
                print(f"エラー詳細 (テキスト): {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"DiscordへのAPIリクエスト中にエラーが発生しました: {e}")
        return False
    except Exception as e:
        print(f"画像送信中に予期せぬエラーが発生しました: {e}")
        return False
    
def main():
    global avg_background
    avg_background = None
    motion_detection_enabled = False

    parser = argparse.ArgumentParser(description="OpenCV Motion Detector")
    parser.add_argument(
        '--gui',
        action='store_true',
        dest='display_gui',
        help="Enable GUI display"
    )
    parser.set_defaults(display_gui=False)
    args = parser.parse_args()

    ensure_dir(SAVE_DIR)

    cap = cv.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("Starting motion detection module.")
    if args.display_gui:
        print(f"GUI display enabled. Press '{chr(KEY_SET_BACKGROUND)}' to set/reset background.")
        print(f"Press 'Esc' to quit.")
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

    # 最後に通知した時刻を記録する変数を初期化
    last_notification_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            dt_now = datetime.datetime.now()
            dt_format_string = dt_now.strftime('%Y-%m-%d %H:%M:%S')
            
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            if avg_background is None and args.display_gui:
                cv.putText(frame, f"Press '{chr(KEY_SET_BACKGROUND)}' to set background", (25, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif avg_background is not None: # 背景が設定されていれば検知処理
                cv.accumulateWeighted(gray, avg_background, ACCUMULATE_WEIGHT)
                frame_delta = cv.absdiff(gray, cv.convertScaleAbs(avg_background))
                thresh = cv.threshold(frame_delta, DOT_TH, DELTA_MAX, cv.THRESH_BINARY)[1]
                motion_factor = thresh.sum() / (thresh.size * DELTA_MAX)
                motion_factor_str = f'{motion_factor:.08f}'
                
                if args.display_gui:
                    cv.putText(frame, motion_factor_str, (25, FRAME_HEIGHT - 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                if motion_detection_enabled and motion_factor > MOTION_FACTOR_TH:
                    # --- クールダウンタイムのチェック ---
                    current_time_seconds = time.time()
                    if (current_time_seconds - last_notification_time) > NOTIFICATION_COOLDOWN_SECONDS:
                        # モーション検知かつクールダウンタイム経過
                        # ファイル名はタイムスタンプをベースにし、設定された接尾辞（拡張子含む）を付加
                        f_name_base = dt_now.strftime('%Y%m%d%H%M%S%f')
                        actual_filename_to_save = f_name_base + "_" + FN_SUFFIX 
                        save_path = os.path.join(SAVE_DIR, actual_filename_to_save)
                        
                        frame_to_save = frame.copy()
                        cv.putText(frame_to_save, dt_format_string, (25, 50),
                                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        
                        if cv.imwrite(save_path, frame_to_save):
                            print(f"DETECTED & SAVED: {actual_filename_to_save} (Factor: {motion_factor_str})")

                            # --- Discordに通知 (API経由) ---
                            # APIに渡すファイル名は、実際に保存したファイル名が良い
                            if send_image_to_discord_api(save_path, actual_filename_to_save):
                                last_notification_time = current_time_seconds # 通知成功なら最終通知時刻を更新
                        else:
                            print(f"エラー: 画像の保存に失敗しました: {save_path}")
                    # else: # クールダウン中は通知をスキップ (ログ出力は任意)
                        # print(f"Motion detected (Factor: {motion_factor_str}) but notification skipped due to cooldown.")

                if args.display_gui:
                    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    cv.drawContours(frame, contours, -1, (0, 255, 0), 2)

            if args.display_gui:
                cv.imshow('Motion Detection Camera', frame)
                key = cv.waitKey(1) & 0xFF
                
                if key == KEY_QUIT:
                    print("Exit key pressed.")
                    break
                elif key == KEY_SET_BACKGROUND:
                    avg_background = gray.copy().astype("float")
                    motion_detection_enabled = True
                    print(f"Background re-set at {dt_format_string}. Motion detection enabled.")
            else:
                time.sleep(HEADLESS_LOOP_DELAY)

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C). Exiting...")
    finally:
        print("Exiting motion detection module.")
        cap.release()
        if args.display_gui:
            cv.destroyAllWindows()

if __name__ == '__main__':
    main()