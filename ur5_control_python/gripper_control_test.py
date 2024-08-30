import socket
import time

class Gripper:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = None

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))
        print(f"Connected to gripper at {self.ip}:{self.port}")

    def disconnect(self):
        if self.socket:
            self.socket.close()
            print("Disconnected from gripper")

    def send_command(self, command):
        if self.socket:
            self.socket.sendall(command.encode())
            response = self.socket.recv(1024).decode()
            print(f"Sent: {command}, Received: {response}")
        else:
            print("Not connected to gripper")

    def open(self):
        self.send_command("OPEN")

    def close(self):
        self.send_command("CLOSE")

    def set_position(self, position):
        self.send_command(f"POSITION {position}")

# 使用示例
if __name__ == "__main__":
    gripper = Gripper("192.168.51.253", 8886)  # 替换为实际的IP和端口
    
    try:
        gripper.connect()
        
        gripper.open()
        time.sleep(1)
        
        gripper.close()
        time.sleep(1)
        
        gripper.set_position(50)  # 设置到50%的位置
        time.sleep(1)
        
    finally:
        gripper.disconnect()