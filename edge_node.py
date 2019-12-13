#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psutil
import time
import datetime
import json
import socketserver
import argparse
import struct
import paramiko
import pickle
import os

import const as C
import certain_amount as edge_reid


raspi = C.raspi


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class EdgeHandler(socketserver.BaseRequestHandler):

    def recv_file(self, filename, file_size):
        f = open(filename, 'wb')
        buffer = b''
        length = int(file_size)
        while(length > 0):
            data = self.request.recv(length)
            if not data:
                return False
            buffer += data
            length = int(file_size) - len(buffer)
        f.write(buffer)
        f.close()
        print('receive file successfully')

    def send_file(self, filename):
        print('start sending file to edge master')
        f = open(filename, 'rb')
        self.request.sendall(f.read())
        f.close()
        # time.sleep(1)
        # self.request.sendall(b'EOF')
        # time.sleep(1)
        print('send file successfully')


    def handle(self):

        data = self.request.recv(1024)
        print(data)
        data_str_array = data.decode().strip().split("#")
        print(data_str_array)
        control_msg = data_str_array[0]
        print(control_msg)
        self.request.sendall(b'success')


        edge_node = EdgeNode('node1', '192.168.1.143')
        # edge_node = EdgeNode('node1', '10.113.160.4')


        if control_msg == 'status':
            # return node status
            print(control_msg)
            node_status = edge_node.get_sys_stats()
            self.send_msg(node_status.encode())


        if control_msg == 'start':
            # 1. receive image
            print('START RECEVING IMAGE')
            img_size = data_str_array[2]
            filename = 'test_query_result.mat'
            self.recv_file(filename, img_size)

            # 2. start raspi
            raspi_ids = data_str_array[3].split('*')
            raspi_ids.pop()
            print('raspi_ids:{}'.format(raspi_ids))
            raspi_ip = []
            raspi_list = []
            for raspi_id in raspi_ids:
                raspi_ip.append(raspi[raspi_id][0])
                raspi_list.append(raspi[raspi_id][1])
            print(raspi_ip, raspi_list)
            edge_reid.start_raspi(raspi_ip)

            # 3. run reid
            self.request.send(b'start counting')
            self.request.recv(15)


            similarity, sim_box, img_name = edge_reid.run(raspi_list, filename, edge_node.node_id)
            print('-----------------------------')
            print('img_name: {}'.format(img_name))

            # 4. return results
            data = pickle.dumps([similarity, sim_box])
            print('data: {}'.format(data))
            self.request.send(data)
            suc0 = self.request.recv(8)
            print('suc0: {}'.format(suc0))
            msg = ''
            for img in img_name:
                img_size = str(os.path.getsize(img))
                print('img_size: {}'.format(img_size))
                msg = msg + img_size + '#'

            self.request.sendall(msg.encode())
            suc1 = self.request.recv(8)
            print('------------------------------------------------------')
            print('------------------------------------------------------')
            print('------------------------------------------------------')
            print('suc1: {}'.format(suc1))
            
            for img in img_name:
                print(img)
                self.send_file(img)
                suc = self.request.recv(8)
                print(suc)

            # self.shutdown()
            

class EdgeNode:

    def __init__(self, node_id, node_ip):
        self.node_id = node_id
        self.node_ip = node_ip

    def get_sys_stats(self):
        available_cpu = None
        available_mem = None
        available_disk = None
        pending_list = None

        # get cpu
        cpu_usage = psutil.cpu_percent(interval=0.1)
        cpu_number = psutil.cpu_count()
        available_cpu = cpu_number * (1 - cpu_usage)

        # get memory
        available_mem = psutil.virtual_memory().available

        # get disk
        available_disk = psutil.disk_usage('/').free

        # get disk io, network io, network connections, etc.

        return available_cpu, available_mem, available_disk, pending_list


    def get_running_tasks(self):
        pass



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('node_id')
    # parser.add_argument('node_ip')
    # args = parser.parse_args()

    HOST = '192.168.1.143'


    PORT = C.EDGE_NODE_PORT
    
    # with ThreadedTCPServer((HOST, PORT), EdgeHandler) as server:
    server = ThreadedTCPServer((HOST, PORT), EdgeHandler)
    server.serve_forever()

