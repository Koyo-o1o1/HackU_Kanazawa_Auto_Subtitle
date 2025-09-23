from app import app # app.pyからFlaskアプリのインスタンスをインポート

if __name__ == '__main__':
    # Flaskの組み込みサーバーを使ってHTTPSを有効にする
    # この方法はローカルでの利用に最適です
    app.run(
        host='0.0.0.0',
        port=5000,
        ssl_context=('cert.pem', 'key.pem') # ← 作成した証明書ファイルを指定
    )