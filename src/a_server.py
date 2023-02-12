#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String
import socket

HOST = '127.0.0.1'  # Standard loopback interface address (localhost) default
PORT = 5000        # Port to listen on (non-privileged ports are > 1023)
BACKLOG = 2 # max num of clients
MAXDATASIZE = 100 # maximum accepted datasize at a time  

global conn, address, from_robot_pub


def myhook():
    global conn
    print("Close the connection")
    conn.close()  # close the connection
    print("Connection closed")

def callback(msg):
    global conn, address, from_robot_pub
    data_to_send = msg.data
    conn.send(data_to_send.encode())  # send data to the client
    print("ho mandato")
    data_received = conn.recv(PORT).decode()
    if not data_received:
        # if data is not received break
        return

    print("from connected user: " + str(data_received))
    if str(data_received) == 'OK' or str(data_received) == 'ok':
        from_robot_pub.publish('ok')
    else:
        from_robot_pub.publish('fail')

 
if __name__ == '__main__':
    try:
        global conn, address, from_robot_pub
        # topic     
        rospy.init_node('server_socket', anonymous=True)
        rospy.on_shutdown(myhook)

        node_name = rospy.get_name()
        to_robot_topic = rospy.get_param(node_name + '/topic_to_robot')
        from_robot_topic = rospy.get_param(node_name + '/topic_from_robot')
        HOST = rospy.get_param(node_name + '/ip')
        PORT = int(rospy.get_param(node_name + '/port'))

        rospy.Subscriber(to_robot_topic, String, callback)     # mi sottoscrivo ai comandi da mandare al robot che ricevo dal nodo di controllo 
        from_robot_pub = rospy.Publisher(from_robot_topic, String, queue_size=10) # pubblico la risposta del robot per il nodo di controllo 

        rate = rospy.Rate(10) # 10hz

        # get host name
        #host = socket.gethostname()
        #port = 5000  # initiate port no above 1024

        server_socket = socket.socket()  # get instance
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # look closely. The bind() function takes tuple as argument
        server_socket.bind((HOST, PORT))  # bind host address and port together

        # configure how many client the server can listen simultaneously
        server_socket.listen(2)
        print("Waiting for connection ...")
        conn, address = server_socket.accept()  # accept new connection
        print("Connection from: " + str(address))
        
        rospy.spin()


    except rospy.ROSInterruptException:
        pass