from math import pi

import urx
import logging
import time
if __name__ == "__main__":
    rob = urx.Robot("192.168.51.254")
    print(rob)
    #rob = urx.Robot("localhost")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))
    try:
        l = 0.10
        v = 0.4
        a = 0.6
        j = rob.getj()
        print("Initial joint configuration is ", j)
        t = rob.get_pose()
        print("Transformation from base to tcp is: ", t)
        print("Translating in x")
        # rob.translate((l, 0, 0), acc=a, vel=v)
        pose = rob.getl()
        print("robot tcp is at: ", pose)
        print("moving in z")

        pose = rob.getl()
        pose[0] -= l
        rob.movel(pose, acc=a, vel=v)
        time.sleep(2)
        pose = rob.getl()
        pose[0] += l
        rob.movel(pose, acc=a, vel=v)
        time.sleep(2)

        pose = rob.getl()
        pose[1] -= l
        rob.movel(pose, acc=a, vel=v)
        time.sleep(2)
        pose = rob.getl()
        pose[1] += l
        rob.movel(pose, acc=a, vel=v)

        pose = rob.getl()
        pose[2] -= l
        rob.movel(pose, acc=a, vel=v)
        time.sleep(2)
        pose = rob.getl()
        pose[2] += l
        rob.movel(pose, acc=a, vel=v)

        # pose = rob.getl()
        # pose[3] -= 0.8
        # rob.movel(pose, acc=a, vel=v)
        # pose[3] += 1.6
        # rob.movel(pose, acc=a, vel=v)
        # pose[3] -= 0.8
        # rob.movel(pose, acc=a, vel=v)


        # pose[4] -= 0.8
        # rob.movel(pose, acc=a, vel=v)
        # pose[4] += 1.6
        # rob.movel(pose, acc=a, vel=v)
        # pose[4] -= 0.8
        # rob.movel(pose, acc=a, vel=v)        
        
        # print("Translate in -x and rotate")
        # t.orient.rotate_zb(pi/4)
        # t.pos[0] -= l
        # rob.set_pose(t, vel=v, acc=a)
        
        # t.pos[0] += l
        # rob.set_pose(t, vel=v, acc=a)
        
        # print("Sending robot back to original position")
        # rob.movej(j, acc=0.8, vel=0.2) 


    finally:
    # import time
    # time.sleep(1)
        rob.close()