import discord
from discord.ext import commands
import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import io # メモリ上でファイルを扱うために必要
import os # 環境変数を読み込むために必要
from dotenv import load_dotenv # .envファイルから環境変数を読み込むために推奨

# .envファイルから環境変数をロード (プロジェクトルートに .env ファイルを置く想定)
# .env ファイルの例:
# DISCORD_BOT_TOKEN="YourActualBotTokenGoesHere"
# TARGET_DISCORD_CHANNEL_ID="123456789012345678"
load_dotenv()

# Botの設定
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
# チャンネルIDも環境変数から読み込むか、ここで直接指定
# TARGET_CHANNEL_ID = int(os.getenv("TARGET_DISCORD_CHANNEL_ID", "0")) # デフォルト値を設定
TARGET_CHANNEL_ID = 1369884823522640012 # または、以前のコードのように直接指定

if not DISCORD_BOT_TOKEN:
    print("エラー: 環境変数 DISCORD_BOT_TOKEN が設定されていません。")
    exit()
if not TARGET_CHANNEL_ID: # もし環境変数から読み込んでいて、設定されていなかった場合のチェック
    print("エラー: 通知先のチャンネルID (TARGET_CHANNEL_ID) が設定されていません。")
    exit()


intents = discord.Intents.default()
# intents.message_content = True # メッセージ内容を読む権限が必要な場合

bot = commands.Bot(command_prefix="!", intents=intents) # コマンドプレフィックスは今のところ使わない
app = FastAPI()

is_bot_ready = False # Botの準備状態フラグ

@bot.event
async def on_ready():
    global is_bot_ready
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    print(f'Targeting channel ID: {TARGET_CHANNEL_ID}')
    channel = bot.get_channel(TARGET_CHANNEL_ID)
    if channel:
        print(f"Found target channel: {channel.name} (ID: {channel.id})")
    else:
        print(f"警告: ターゲットチャンネル (ID: {TARGET_CHANNEL_ID}) が見つかりません。Botがそのサーバーに参加しているか、IDが正しいか確認してください。")
    is_bot_ready = True
    print('Discord Bot is ready.')

# 画像送信用のAPIエンドポイント
@app.post("/send_image/")
async def send_image_api(file: UploadFile = File(...)):
    if not is_bot_ready:
        # Botが準備できていない場合は503 Service Unavailableエラーを返す
        raise HTTPException(status_code=503, detail="Discord Botの準備ができていません。少し待ってから再試行してください。")

    channel = bot.get_channel(TARGET_CHANNEL_ID)
    if channel is None:
        # チャンネルが見つからない場合は500 Internal Server Error (または404 Not Foundでも良いが、サーバー内部の問題として500が適切か)
        print(f"エラー: APIリクエスト時、ターゲットチャンネル (ID: {TARGET_CHANNEL_ID}) が見つかりませんでした。")
        raise HTTPException(status_code=500, detail=f"サーバーエラー: Discordチャンネル (ID: {TARGET_CHANNEL_ID}) が見つかりません。")

    try:
        contents = await file.read() # ファイル内容をメモリに読み込む
        if not contents:
            raise HTTPException(status_code=400, detail="アップロードされたファイルが空です。")

        # discord.File には BytesIOオブジェクトとファイル名を渡す
        discord_file = discord.File(io.BytesIO(contents), filename=file.filename or "image.png")
        
        await channel.send(file=discord_file)
        print(f"画像 '{file.filename or 'image.png'}' をチャンネル '{channel.name}' に送信しました。")
        return {"status": "success", "message": f"画像をチャンネル '{channel.name}' に送信しました"}

    except discord.errors.Forbidden:
        print(f"エラー: チャンネル '{channel.name}' への送信権限がありません。")
        raise HTTPException(status_code=403, detail=f"Discordエラー: チャンネル '{channel.name}' への送信権限がありません。")
    except discord.errors.HTTPException as e:
        print(f"エラー: Discord APIへの送信中にHTTPエラーが発生しました: {e.status} - {e.text}")
        raise HTTPException(status_code=502, detail=f"Discord APIエラー: {e.text} (ステータス: {e.status})")
    except Exception as e:
        # その他の予期せぬエラー
        print(f"エラー: 画像送信中に予期せぬエラーが発生しました: {e}")
        # スタックトレースもログに出力するとデバッグに役立つ
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"サーバー内部エラー: {str(e)}")

# Bot起動用非同期関数
async def main_async_runner():
    # FastAPIサーバー起動（uvicorn）
    # ホストを "127.0.0.1" にすることで、同じコンピュータからのアクセスのみ許可する
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)

    # BotとAPIサーバーを並行で動かす
    # bot.start() はブロッキングなので、asyncio.create_task でラップして await しないか、
    # bot.login() と bot.connect() を使うなど、より高度な制御が必要な場合もあるが、
    # asyncio.gather はそれぞれのコルーチンが完了するまで待つ。
    try:
        print("Starting Discord Bot and FastAPI server...")
        await asyncio.gather(
            bot.start(DISCORD_BOT_TOKEN), # bot.start は内部でループを持つ
            server.serve(),               # server.serve も内部でループを持つ
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt MROOOOOO received. Shutting down...")
    finally:
        print("Closing Discord Bot connection...")
        if bot.is_ready(): # is_ready() は on_ready より前に呼ばれると False の可能性
            await bot.close() # Botを安全に閉じる
        print("FastAPI server shutdown (Uvicorn handles its own shutdown on Ctrl+C usually)...")
        # UvicornのServerオブジェクトには明示的なstopメソッドがない場合があるが、
        # server.should_exit フラグを立てるなどして外部から停止させることは可能。
        # asyncio.gatherがキャンセルされると、server.serve()も終了するはず。

if __name__ == "__main__":
    try:
        asyncio.run(main_async_runner())
    except KeyboardInterrupt:
        print("Application shutdown initiated by KeyboardInterrupt in __main__.")
    finally:
        print("Application has finished.")