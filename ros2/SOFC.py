import rclpy,os
from pymycobot.mycobot import MyCobot
from rclpy.node import Node
from mecharm_interfaces.msg import MecharmCoords
from std_msgs.msg import String
from pymycobot import MyCobotSocket
import torch as th
from ament_index_python import get_package_share_directory

from stable_baselines3.common.torch_layers import create_mlp
import numpy as np
import tkinter as tk
import time
from datetime import datetime

class SOFC_policy(th.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        actor_net = create_mlp(3, 6, [64,64], 
                         th.nn.ReLU, squash_output=True)
        self.actor_net = th.nn.Sequential(*actor_net)
        
    def forward(self, noisy_outputs:th.Tensor):
        return self.actor_net(noisy_outputs)
        
    def load(self,path:str)->None:
        vector = np.load(path)
        self.load_from_vector(vector)

    def load_from_vector(self, vector: np.ndarray) -> None:
        """
        Load parameters from a 1D vector.

        :param vector:
        """
        th.nn.utils.vector_to_parameters(th.as_tensor(vector, dtype=th.float, device=th.device("cpu")), self.parameters())

class SOFC(Node):
    """Static Output Feedback Control Policy based on PyTorch"""
    def __init__(self):
        super().__init__("SOFC")
        
        self.CTL = "None"
        self.cnt = 0 
        self.x=[]
        self.data = np.ones(7)
        
        self.sofc_kde = SOFC_policy()
        path = os.path.join(
            get_package_share_directory("mecharm_pi"),
            "config/KDE_SOFC.npy"
        )
        self.sofc_kde.load(path)
        
        self.sofc_maf = SOFC_policy()
        path = os.path.join(
            get_package_share_directory("mecharm_pi"),
            "config/MAF_SOFC.npy"
        )
        self.sofc_maf.load(path)
        
        self.sub_button = self.create_subscription(
            String,
            "button_event",
            self.button_callback,
            10
        )
        
        self.sub_noisy_outputs = self.create_subscription(
            MecharmCoords,
            "noisy_outputs",
            self.ctl_callback,
            10
        )

        self.mc = MyCobot("/dev/ttyAMA0", 1000000)


    def button_callback(self, msg):
        self.CTL = msg.data
        name = datetime.now().strftime("%m%d%H%M%S")
        self.get_logger().info("status[{}]: {}".format(name,self.CTL))
        self.x = []
        while self.x == []:
              self.x = self.mc.get_radians()
        if(msg.data== "save"):
            np.save("/home/er/data/data_{}".format(name),self.data)
            self.data=np.ones(7)
            self.cnt=0
        
    
    def ctl_callback(self,msg):
        isCtl = True
        inputs=[]
        noisy_outputs = th.tensor([msg.x,msg.y,msg.z],dtype=th.float32)
        if self.CTL == "KDE":
            inputs = self.sofc_kde(noisy_outputs).tolist()
        elif self.CTL == "MAF":
            inputs = self.sofc_maf(noisy_outputs).tolist()
        else:
            isCtl=False
        if isCtl:
            if self.cnt % 10 == 0: #calibration
                self.x=[]
                while self.x == []:
                    self.x = self.mc.get_radians()
            for i in range(6):
                self.x[i]=self.x[i]+0.05*inputs[i]
            self.mc.send_radians(self.x, 40)
            
            data= [msg.x,msg.y,msg.z,msg.rx,msg.ry,msg.rz,np.linalg.norm(inputs)]
            self.data= np.vstack([self.data,data])
            self.cnt += 1
            self.get_logger().info("status[{}]: {}".format(self.cnt,[msg.rx,msg.ry,msg.rz]))
            

def main(args=None):
    rclpy.init(args=args)
    noisy_outputs_subscriber = SOFC() 
    #the SOFC node subscribes the noisy outputs to generate the input signal
    
    rclpy.spin(noisy_outputs_subscriber)
    
    noisy_outputs_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
