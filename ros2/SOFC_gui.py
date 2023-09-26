import rclpy,os
from pymycobot.mycobot import MyCobot
from rclpy.node import Node
from std_msgs.msg import String
from pymycobot import MyCobotSocket
import torch as th
from ament_index_python import get_package_share_directory

from stable_baselines3.common.torch_layers import create_mlp
import numpy as np
import tkinter as tk
import time

class Window: 
    def __init__(self, handle):
        self.mc = MyCobot("/dev/ttyAMA0", 1000000)
        
        
        self.win = handle
        self.win.resizable(0, 0) 
        
        button_node = Node("button")
        self.pub = button_node.create_publisher(String, "button_event",10)


        # get screen width and height
        self.ws = self.win.winfo_screenwidth()  # width of the screen
        self.hs = self.win.winfo_screenheight()  # height of the screen
        
        # calculate x and y coordinates for the Tk root window
        x = (self.ws / 2) - 190
        y = (self.hs / 2) - 350
        self.win.geometry("450x420+{}+{}".format(int(x), int(y)))

        self.frmTop = tk.Frame(width=200, height=100)
        self.frmBot = tk.Frame(width=200, height=100)
        self.frmTop.grid(row=0, column=0, padx=1, pady=3)
        self.frmBot.grid(row=1, column=0, padx=1, pady=3)

        tk.Button(self.frmTop, text="reset", width=5, command=self.reset).grid(
            row=0, column=0, sticky="w", padx=3, pady=2
        )

        tk.Button(self.frmTop, text="neutral", width=5, command=self.neutral).grid(
            row=0, column=1, sticky="w", padx=3, pady=2
        )

        tk.Button(self.frmTop, text="release", width=5, command=self.release).grid(
            row=0, column=2, sticky="w", padx=3, pady=2
        )
        tk.Button(self.frmBot, text="KDE_SOFC", command=self.KDE, width=10).grid(
            row=1, column=0, sticky="w", padx=3, pady=2
        )
        tk.Button(self.frmBot, text="MAF_SOFC", command=self.MAF, width=10).grid(
            row=1, column=1, sticky="w", padx=3, pady=2
        )
        tk.Button(self.frmBot, text="sim_pos", command=self.sim_pos, width=10).grid(
            row=2, column=1, sticky="w", padx=3, pady=2
        )  
        tk.Button(self.frmBot, text="Save", command=self.save, width=10).grid(
            row=2, column=0, sticky="w", padx=3, pady=2
        )        
        
    def reset(self):
        msg =String()
        msg.data = "reset"
        self.pub.publish(msg)
        tk.Label(self.frmBot, text=msg.data).grid(row=0)
        x0 = np.random.uniform(-0.8,0.8,size=(6,)).tolist()
        self.mc.send_radians(x0, 30)
            
            
    def sim_pos(self):
        msg =String()
        msg.data = "simulation"
        self.pub.publish(msg)
        tk.Label(self.frmBot, text=msg.data).grid(row=0)
        self.mc.send_radians([0.5,-1,-0.5,0.5,0.5,0], 30)        
            
    def neutral(self):
        msg =String()
        msg.data = "neutral"
        self.pub.publish(msg)
        tk.Label(self.frmBot, text=msg.data).grid(row=0)
        self.mc.send_radians([0,0,0,0,0,0], 30)
        
    def release(self):
        msg =String()
        msg.data = "release"
        self.pub.publish(msg)
        tk.Label(self.frmBot, text=msg.data).grid(row=0)
        self.mc.release_all_servos()
        
    def KDE(self):
        msg =String()
        msg.data = "KDE"
        self.pub.publish(msg)
        tk.Label(self.frmBot, text=msg.data).grid(row=0)

            
    def MAF(self):
        msg =String()
        msg.data = "MAF"
        self.pub.publish(msg)
        tk.Label(self.frmBot, text=msg.data).grid(row=0)
        
                
    def save(self):
        msg =String()
        msg.data = "save"
        self.pub.publish(msg)
        tk.Label(self.frmBot, text=msg.data).grid(row=0)
        
    def run(self):
        while True:
            try:
                self.win.update()
                # print("ok")
                time.sleep(0.01)
            except tk.TclError as e:
                if "application has been destroyed" in str(e):
                    break
                else:
                    raise        

def main(args=None):
    rclpy.init(args=args)

    window = tk.Tk()
    window.title("DE-SAC with mechArm270")
    Window(window).run()
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()
