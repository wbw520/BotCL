from flask import Flask


app = Flask(__name__)


@app.route('/')
def index():
    return {
        "msg": "success",
        "data": "welcome to use flask."
    }


if __name__ == '__main__':
    app.debug = False # 设置调试模式，生产模式的时候要关掉debug
    app.run()