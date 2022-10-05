from web import app
from web.socket import socket_init, callback
from core import video_process
import threading

if __name__ == '__main__':
    # 新开两个线程，一个用于转换函数，另外一个为websocket服务
    t1 = threading.Thread(target=video_process, args=(callback,))
    t2 = threading.Thread(target=socket_init)
    t1.start()
    t2.start()
    # 运行flask
    app.run(host='0.0.0.0', port=7001)
