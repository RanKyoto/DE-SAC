import rclpy
from pymycobot.mycobot import MyCobot
from rclpy.node import Node
from mecharm_interfaces.msg import MecharmCoords
import numpy as np

class Noisy_Sensor(Node):
     def __init__(self):
            super().__init__("follow_display")

            self.mc = MyCobot("/dev/ttyAMA0", 1000000)
            
     def start(self):
        self.pub = self.create_publisher(
            msg_type=MecharmCoords,
            topic="noisy_outputs",
            qos_profile=1
        )
        self.mc.send_radians([0,0,0,0,0,0],20)
        self.timer = self.create_timer(0.2,self.timer_callback)
        
     def timer_callback(self):
        coords = MecharmCoords()
        data_list = []
        try:
            output = self.mc.get_coords()[0:3] #mm
            for _, value in enumerate(output):
                data_list.append(value/1000+np.random.uniform(-1,1)*0.05)
                #data_list.append(value/1000+np.random.randn()*0.05)
            coords.x = data_list[0] + 0.0153
            coords.y = data_list[1]
            coords.z = data_list[2] + 0.0217
            coords.rx = output[0]
            coords.ry = output[1]
            coords.rz = output[2]
        
            self.pub.publish(coords)
        except Exception:
            pass
        
def main(args=None):   
    rclpy.init(args=args)
    
    sensor = Noisy_Sensor()
    sensor.start()
    rclpy.spin(sensor)     

if __name__ == "__main__":
    main()

