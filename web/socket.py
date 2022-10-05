import asyncio
import websockets
# https://websockets.readthedocs.io/en/stable/

# 全局websocket服务
_ws = None


# 回调函数
def callback(data):
    if _ws:
        asyncio.run(send_message(data))


# 异步消息发送函数
async def send_message(data):
    await _ws.send(data)


# websocket服务
async def echo(websocket):
    global _ws
    _ws = websocket
    async for message in websocket:
        await websocket.send(message)
    # websocket关闭后置为空
    _ws = None


# 启动websock服务
async def start_server():
    async with websockets.serve(echo, "0.0.0.0", 7002):
        await asyncio.Future()  # run forever


# 初始化websocket服务
def socket_init():
    asyncio.run(start_server())
