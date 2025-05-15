# -*- coding: utf-8 -*-
import time
import datetime
import cv2 as cv
import os
import argparse
import requests


#https://qiita.com/hase-k0x01/items/acd0d9159a9001ebfbd3

# --- 設定項目 ---
SAVE_DIR = './video_clips/' # 動画クリップの保存ディレクトリ
CLIP_FN_SUFFIX = 'motion_clip.mp4' # 動画ファイル名の接尾辞
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_RECORD = 10 # 録画する動画のフレームレート 
RECORDING_DURATION_SECONDS = 5 # 1回のモーション検知で録画する時間（秒）

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

def send_file_to_discord_api(file_path: str, filename_on_discord: str, mime_type: str = "video/mp4"):
    """
    指定されたファイルをFastAPI経由でDiscordに送信する関数。
    静止画の場合は mime_type="image/jpeg" などとする。
    """
    if not os.path.exists(file_path):
        print(f"エラー: 通知するファイルが見つかりません: {file_path}")
        return False
    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename_on_discord, f, mime_type)}
            # 動画ファイルは大きくなる可能性があるのでタイムアウトを少し長めに
            response = requests.post(API_ENDPOINT_URL, files=files, timeout=30)

        if response.status_code == 200:
            print(f"Discordへのファイル送信成功: {response.json().get('message', '成功')}")
            return True
        else:
            print(f"Discordへのファイル送信失敗: ステータスコード {response.status_code}")
            try:
                print(f"エラー詳細: {response.json()}")
            except requests.exceptions.JSONDecodeError:
                print(f"エラー詳細 (テキスト): {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"DiscordへのAPIリクエスト中にエラーが発生しました: {e}")
        return False
    except Exception as e:
        print(f"ファイル送信中に予期せぬエラーが発生しました: {e}")
        return False
    
def main():
    global avg_background
    avg_background = None
    motion_detection_enabled = False

    parser = argparse.ArgumentParser(description="OpenCV Motion Detector with Video Clips")
    parser.add_argument('--gui', action='store_true', dest='display_gui', help="Enable GUI display")
    parser.set_defaults(display_gui=False)
    args = parser.parse_args()

    ensure_dir(SAVE_DIR) # 動画保存用ディレクトリ

    cap = cv.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


    print("Starting motion detection module.")
    if not args.display_gui: # ヘッドレスモード時の初期背景設定
        print("Setting initial background automatically...")
        ret, frame_for_bg = cap.read()
        if ret:
            gray_for_bg = cv.cvtColor(frame_for_bg, cv.COLOR_BGR2GRAY)
            avg_background = gray_for_bg.copy().astype("float")
            motion_detection_enabled = True
            print(f"Initial background set. Motion detection enabled.")
        else:
            print("Error: Could not read frame for initial background. Exiting.")
            cap.release()
            return

    last_notification_time = 0
    
    is_recording = False # 現在録画中かどうか
    recording_start_time = 0
    video_writer = None
    current_clip_path = None
    current_clip_filename = None
        # 動画コーデックの指定 (MP4の場合)
    # fourcc = cv.VideoWriter_fourcc(*'mp4v') # .mp4
    fourcc = cv.VideoWriter_fourcc(*'XVID') # .avi (より確実性が高い場合があるが、ファイルサイズは大きめになるかも)
                                           # XVID を使う場合は、FN_SUFFIX を .avi に変更推奨

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                break

            dt_now = datetime.datetime.now()
            dt_format_string = dt_now.strftime('%Y-%m-%d %H:%M:%S')
            
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if avg_background is None and args.display_gui:
                cv.putText(frame, f"Press '{chr(KEY_SET_BACKGROUND)}' to set background", (25, 80),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            elif avg_background is not None:
                cv.accumulateWeighted(gray, avg_background, ACCUMULATE_WEIGHT)
                frame_delta = cv.absdiff(gray, cv.convertScaleAbs(avg_background))
                thresh = cv.threshold(frame_delta, DOT_TH, DELTA_MAX, cv.THRESH_BINARY)[1]
                motion_factor = thresh.sum() / (thresh.size * DELTA_MAX)
                motion_factor_str = f'{motion_factor:.08f}'

                if args.display_gui:
                    cv.putText(frame, motion_factor_str, (25, FRAME_HEIGHT - 20),
                               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # --- モーション検知 & 録画 & 通知処理 ---
                current_time_seconds = time.time()
                if motion_detection_enabled and motion_factor > MOTION_FACTOR_TH and \
                   not is_recording and \
                   (current_time_seconds - last_notification_time) > NOTIFICATION_COOLDOWN_SECONDS:
                    
                    print(f"Motion DETECTED! Factor: {motion_factor_str}. Starting video recording...")
                    is_recording = True
                    recording_start_time = current_time_seconds
                    
                    f_name_base = dt_now.strftime('%Y%m%d%H%M%S%f')
                    # 動画ファイル名を生成 (fourcc に合わせて拡張子を .avi に変更も検討)
                    current_clip_filename = f_name_base + "_" + ("motion_clip.avi" if fourcc == cv.VideoWriter_fourcc(*'XVID') else CLIP_FN_SUFFIX)
                    current_clip_path = os.path.join(SAVE_DIR, current_clip_filename)
                    
                    # VideoWriterオブジェクトを作成
                    # サイズは (FRAME_WIDTH, FRAME_HEIGHT) で指定
                    video_writer = cv.VideoWriter(current_clip_path, fourcc, FPS_RECORD, (FRAME_WIDTH, FRAME_HEIGHT))
                    
                    if not video_writer.isOpened():
                        print(f"エラー: VideoWriterのオープンに失敗しました。パス: {current_clip_path}")
                        is_recording = False # 録画フラグをリセット
                        video_writer = None
                    else:
                        print(f"Recording started: {current_clip_filename}")


                # 録画中の処理
                if is_recording and video_writer is not None:
                    video_writer.write(frame) # 現在のフレームを動画に書き込む
                    
                    # 録画時間が経過したかチェック
                    if (current_time_seconds - recording_start_time) >= RECORDING_DURATION_SECONDS:
                        print(f"Recording finished: {current_clip_filename}")
                        video_writer.release()
                        video_writer = None
                        is_recording = False # 録画フラグをリセット
                        
                        # --- Discordに動画クリップを通知 ---
                        print(f"Sending video clip to Discord: {current_clip_filename}")
                        # MIMEタイプを動画用に指定 (例: "video/x-msvideo" for AVI, "video/mp4" for MP4)
                        mime = "video/x-msvideo" if fourcc == cv.VideoWriter_fourcc(*'XVID') else "video/mp4"
                        if send_file_to_discord_api(current_clip_path, current_clip_filename, mime_type=mime):
                            last_notification_time = current_time_seconds # 通知成功でクールダウン開始
                        
                        current_clip_path = None # パスをクリア
                        current_clip_filename = None


                if args.display_gui: # GUI表示中の輪郭描画
                    contours, _ = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                    cv.drawContours(frame, contours, -1, (0, 255, 0), 2)
            
            # --- GUI表示とキー入力処理 ---
            if args.display_gui:
                cv.imshow('Motion Detection Camera', frame)
                key = cv.waitKey(1) & 0xFF
                
                if key == KEY_QUIT:
                    print("Exit key pressed.")
                    if is_recording and video_writer is not None: # 終了時に録画中なら保存
                        video_writer.release()
                        print(f"Recording stopped due to exit: {current_clip_filename if current_clip_filename else 'N/A'}")
                    break
                elif key == KEY_SET_BACKGROUND:
                    avg_background = gray.copy().astype("float")
                    motion_detection_enabled = True
                    print(f"Background re-set at {dt_format_string}. Motion detection enabled.")
            else: # ヘッドレスモード
                if is_recording: # 録画中は少し高頻度でループ (FPS_RECORDに近づけるため)
                    time.sleep(1.0 / (FPS_RECORD * 2)) # 録画FPSの半分の周期でスリープ (CPU負荷とのバランス)
                else:
                    time.sleep(HEADLESS_LOOP_DELAY)

    except KeyboardInterrupt:
        print("Interrupted by user (Ctrl+C). Exiting...")
        if is_recording and video_writer is not None: # Ctrl+C時も録画中なら保存
            video_writer.release()
            print(f"Recording stopped due to interrupt: {current_clip_filename if current_clip_filename else 'N/A'}")
    finally:
        print("Exiting motion detection module.")
        cap.release()
        if video_writer is not None and video_writer.isOpened(): # 念のため確認
            video_writer.release()
        if args.display_gui:
            cv.destroyAllWindows()

if __name__ == '__main__':
    main()