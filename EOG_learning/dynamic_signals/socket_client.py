import socket
import sys

my_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
ip = '127.0.0.1'
port = 8888
ip_address = (ip, port)

while True:
    send_data = input('Type the data you want to send =>')
    my_socket.sendto(send_data.encode('utf-8'),ip_address)