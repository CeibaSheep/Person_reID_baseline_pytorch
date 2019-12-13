#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import psutil
import socketserver
import threading
import logging
import socket
import struct
import time
import os
import pickle

import const as C
import reid_track_single_raspi as edge_reid_single

# from edge_node import EdgeNode


raspi = C.raspi
reid_result = {}



class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


class EdgeMasterSocket:

    def __init__(self, edge_node_ip):
        logging.info('Edge Node IP: %s' % edge_node_ip)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


    def send_file(self, filename):
        print('start sending file to edge node')
        f = open(filename, 'rb')
        self.socket.sendall(f.read())
        f.close()
        # time.sleep(1)
        # self.socket.sendall(b'EOF')
        # time.sleep(1)
        print('send file susccessfully')

    def recv_file2(self, filename):
        print('start receiving file')
        f = open(filename, 'wb')
        while True:
            data = self.socket.recv(4096)
            if data == b'EOF':
                print('receive file successfully')
                break
            f.write(data)
        f.close()

    def recv_file(self, filename, file_size):
        f = open(filename, 'wb')
        buffer = b''
        length = int(file_size)
        while(length > 0):
            data = self.socket.recv(length)
            if not data:
                return False
            buffer += data
            length = int(file_size) - len(buffer)
        print('--------------------------------')
        # print(buffer) 
        f.write(buffer)
        f.close()
        print('receive file successfully')
    

class EdgeMaster:
    
    def __init__(self):
        self.edge_node_list = self.get_edge_node_list()
        self.raspi_list = self.get_raspi_list()
        self.raspi_ip = self.get_raspi_ip()

    # ---- status-related functions ----
    def get_edge_node_list(self):
        # v1.0 init node list manually
        self.edge_node_list = C.node_list
        # TODO v2.0: init node list automatically
        return self.edge_node_list

    def get_raspi_list(self):
        self.raspi_list = C.raspi_list
        return self.raspi_list

    def get_raspi_ip(self):
        self.raspi_ip = C.raspi_ip
        return self.raspi_ip

    def get_edge_node_status(self, node_id):
        node_status = {}
        return node_status
    
    def get_input_image(self):
        filename = 'query_result.mat'
        return filename

    

# send images to corresponding node and receive results
def process_reid(node_id, raspi_list):
    HOST = edge_node_list[node_id]
    PORT = C.EDEG_NODE_PORT
    # Localhost = '192.168.1.243'
    cur_thread = threading.currentThread()

    master_socket = EdgeMasterSocket(HOST)
    # master_socket = EdgeMasterSocket(Localhost)


    master_socket.socket.connect((HOST, PORT))
    # 1. send img to edge node
    img_name = 'query_result.mat'
    # img_name = 'test.jpeg'
    img_size = str(os.path.getsize(img_name))
    msg = 'start#' + img_name + '#' + img_size + '#' + ''.join(raspi_id + '*' for raspi_id in raspi_list)
    print('msg:{}'.format(msg))
    master_socket.socket.sendall(msg.encode())
    # data = master_socket.socket.recv(10)

    # print('Received', str(data))

    data = master_socket.socket.recv(10)
    print(data)
    master_socket.send_file(img_name)

    # wait for result
    count0 = master_socket.socket.recv(15)
    t0 = time.time()
    master_socket.socket.send(b'start counted')



    # sim result
    data = master_socket.socket.recv(4096)
    print(data)
    data_load = pickle.loads(data)
    reid_result[node_id] = data_load
    master_socket.socket.send(b'success0')

    # img size
    img_size_list = master_socket.socket.recv(1024)
    print('------------------------------------------------------')
    print('------------------------------------------------------')
    print('------------------------------------------------------')
    print('{}: img_size_list: {}'.format(cur_thread, img_size_list))
    print('------------------------------------------------------')
    print('------------------------------------------------------')
    print('------------------------------------------------------')
    img_size_split = img_size_list.decode().strip().split('#')
    img_size_split.pop()
    print('{}: img_size: {}'.format(cur_thread, img_size_split))
    master_socket.socket.send(b'success1')
    
    for i, img in enumerate(img_size_split):
        filename = raspi_list[i] + '.jpeg'
        # filename = 'new' + node_id + 'raspi' + str(i) + '.jpeg'
        master_socket.recv_file(filename, img)
        master_socket.socket.send(b'success2')

    t1 = time.time()
    print('time: {}'.format(t1 - t0))
    

if __name__ == '__main__':
    master = EdgeMaster()
    # init
    # node ip address
    edge_node_list = master.get_edge_node_list()
    print('Edge Node List: {}'.format(edge_node_list))

    # for node in edge_node_list:
    #     master.send_raspi_list_to_all_nodes(raspi_list)

    # --------  ReID Process  --------

    # 1. get input image (name)
    input_image = master.get_input_image()

    # 2. get node status and make scheduling decision (skipped)
    # edge_node_status_list = []
    # for node in edge_node_list:
    #     edge_node_status_list.append(master.get_edge_node_status(node.node_id))

    # make a schedule decision according to edge node status
    # random algorithm
    reid_assign = {}
    reid_assign['node1'] = ['raspi1', 'raspi2']
    # reid_assign['node2'] = ['raspi3', 'raspi4']

    # 3. for each node do reid

    thread_list = []
    for node_id in reid_assign.keys():
        # if node_id == 'node1':
        #     continue   
        reid_thread = threading.Thread(target=process_reid, args=(node_id, reid_assign[node_id]))
        reid_thread.start()
        thread_list.append(reid_thread)
    for reid_thread in thread_list:
        reid_thread.join()
    # process_reid('node1', reid_assign['node1'])

    # 4. get reid result
    print(reid_result)
    node_id_sorted = sorted(edge_node_list.keys())
    raspi_similarity = {}
    raspi_sim_box = {}
    for node_idx in node_id_sorted:
        # if node_idx == 'node1':
        #     continue
        for i, raspi_idx in enumerate(reid_assign[node_idx]):
            raspi_similarity[raspi_idx] = reid_result[node_idx][0][i]
            raspi_sim_box[raspi_idx] = reid_result[node_idx][1][i]
    raspi_result_id = max(raspi_similarity, key=raspi_similarity.get)
    print(raspi_similarity)
    print(raspi_result_id)

    # 5. print result
    # raspi_keys = raspi.keys() //liyang

    raspi_keys = ['raspi1', 'raspi2']
    max_sim_frame = [raspi_idx + '.jpeg' for raspi_idx in raspi_keys]
    max_sim_box = [raspi_sim_box[i] for i in raspi_keys]
    max_similarity = [raspi_similarity[i] for i in raspi_keys]

    edge_reid_single.show_result(max_sim_frame, [edge_reid_single.boxes_transform(i) for i in max_sim_box], max_similarity)



    # 6. run single raspi to do the tracking
    server_ip = [raspi[raspi_result_id][0]]
    edge_reid_single.start_raspi(server_ip)
    server_list = [raspi[raspi_result_id][1]]
    edge_reid_single.start_track_single_edge(server_list,input_image)





    

















    




