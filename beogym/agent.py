import random

class Agent():

    
    def __init__(self, data_helper, turning_range, view_resolution, pano_mode):
        
        self.dh = data_helper
        self.turning_range = turning_range
        self.view_res = view_resolution
        self.pano_mode = pano_mode
        self.curr_camera=1
        self.dis_to_goal=0
        # Maps actions to discreet angles
        self.action_angle_map = {
            2: 72,
            3: 144,
            4: 216,
            5: 288,
            6: 360,
        }
        self.past_view = []
        self.reset()
    
    def reset(self,pos=None):
        self.agent_pos_curr,long = self.dh.reset(pos)
        self.source_pos = self.agent_pos_curr
        self.agent_pos_prev = self.agent_pos_curr
        # self.curr_image_name = self.dh.image_name(self.agent_pos_curr)
        # print("Image name: ", self.curr_image_name)
        self.curr_angle = random.randint(0,359)
        if self.pano_mode:
            self.curr_view = self.dh.panorama_split(self.curr_angle, self.agent_pos_curr, self.view_res, True)
        else:
            self.curr_view = self.dh.get_image_orientation(self.agent_pos_curr, self.curr_camera)

        return self.curr_view,long

        # The behaviour of go_back: go straight back but keep looking at the same angle. Similar to google maps.

    # @profile(precision=5)
    def calculate_new_camera_orientation(self, orientation_value):
        """
        Calculates new camera orientation in a clockwise fashion. Going right is +1 and left is -1.
        If camera 0 or 5 is reached, the values continue in a circular fashion.

        Example: If going left from 1, it should be 1->5. If going right from 5, it should go from 5->0

        orientation_value = +1 (for right) or -1 (for left).
        """
        if self.curr_camera == 1 and orientation_value == -1:
            self.curr_camera = 5
        elif self.curr_camera == 5 and orientation_value == 1:
            self.curr_camera = 1
        else:
            self.curr_camera += orientation_value


    # @profile(precision=5)
    def go_back(self):
        # print("\n")
        # print("taking a step back to previous position: ", self.agent_pos_prev)
        new_pos, curr_pos, new_angle = self.dh.find_nearest(self.agent_pos_curr, self.agent_pos_prev, self.curr_angle, "backward")
        self.update_agent(new_pos, curr_pos, new_angle)

    # @profile(precision=5)
    def go_straight(self):

        # print("\n")
        # print("taking a step straight from my current position: ", self.agent_pos_curr)
        new_pos, curr_pos, new_angle = self.dh.find_nearest(self.agent_pos_curr, self.agent_pos_prev, self.curr_angle, "forward")
        self.update_agent(new_pos, curr_pos, new_angle)


    
    def go_left(self):
        if self.pano_mode:
            new_angle = self.dh.fix_angle(self.curr_angle - self.turning_range)
            # curr_image = self.dh.image_name(self.agent_pos_curr)
            self.curr_angle = new_angle
            del self.curr_view
            self.curr_view = self.dh.panorama_split(new_angle, self.agent_pos_curr, self.view_res)
        else:
            # curr_image = self.dh.image_name(self.agent_pos_curr)
            self.calculate_new_camera_orientation(orientation_value=-1)
            self.curr_angle = self.dh.camera_angle_map[self.curr_camera]
            self.curr_view = self.dh.get_image_orientation(self.agent_pos_curr, self.curr_camera)



    # @profile(precision=5)
    def go_right(self):
        if self.pano_mode:
            # curr_image = self.dh.image_name(self.agent_pos_curr)
            new_angle = self.dh.fix_angle(self.curr_angle + self.turning_range)
            self.curr_angle = new_angle
            self.curr_view = self.dh.panorama_split(new_angle,  self.agent_pos_curr, self.view_res)

        else:
            # curr_image = self.dh.image_name(self.agent_pos_curr)
            self.calculate_new_camera_orientation(orientation_value=1)
            self.curr_angle = self.dh.camera_angle_map[self.curr_camera]
            self.curr_view = self.dh.get_image_orientation(self.agent_pos_curr, self.curr_camera)
    # @profile(precision=5)
    def pano_fixed_angle_turn(self, angle):
        if not self.pano_mode:
            raise EnvironmentError("This method should only be used in pano mode.")
        # curr_image = self.dh.image_name(self.agent_pos_curr)
        new_angle = self.dh.fix_angle(angle)
        self.curr_angle = new_angle
        self.curr_view = self.dh.panorama_split(new_angle,  self.agent_pos_curr, self.view_res)

    # Update the current status of the agent:
    # @profile(precision=5)
    def update_agent(self, new_pos, curr_pos, new_angle):
        
        self.agent_pos_prev = self.agent_pos_curr
        self.agent_pos_curr = new_pos
        self.curr_angle = new_angle

        if self.agent_pos_prev==self.agent_pos_curr:
            move = False
        else:
            move = True

        if self.pano_mode:
            self.curr_view = self.dh.panorama_split(self.curr_angle, new_pos, self.view_res,move)
        else:
            self.curr_view = self.dh.get_image_orientation(self.agent_pos_curr, self.curr_camera)


    # @profile(precision=5)
    def take_action(self, action):

        if action == 0:
            self.go_straight()
        elif action == 1:
            new_angle = self.dh.fix_angle(self.curr_angle + 10)
            self.curr_angle = new_angle
            self.curr_view = self.dh.panorama_split(new_angle,  self.agent_pos_curr, self.view_res)
        elif action == 2:
            new_angle = self.dh.fix_angle(self.curr_angle -10)
            self.curr_angle = new_angle
            self.curr_view = self.dh.panorama_split(new_angle,  self.agent_pos_curr, self.view_res)
        elif action == 3:
            new_angle = self.dh.fix_angle(self.curr_angle + 20)
            self.curr_angle = new_angle
            self.curr_view = self.dh.panorama_split(new_angle,  self.agent_pos_curr, self.view_res)
        elif action == 4:
            new_angle = self.dh.fix_angle(self.curr_angle - 20)
            self.curr_angle = new_angle
            self.curr_view = self.dh.panorama_split(new_angle,  self.agent_pos_curr, self.view_res)
        else:
            print('error')
