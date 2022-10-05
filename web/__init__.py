import time

from flask import Flask, request, Response, render_template
from flask_cors import CORS

# 初始化flaskAPP
app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
# 允许跨域请求
CORS(app, resources=r'/*')


# 主页显示HTML
@app.route('/', methods=['GET'])
def index():
    return render_template('content.html')
