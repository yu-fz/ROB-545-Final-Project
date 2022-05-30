import sys
sys.path.append('./underactuated')
import numpy as np 
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, eq, SnoptSolver
from underactuated import running_as_notebook
from underactuated.jupyter import AdvanceToAndVisualize

class SLIP_Model():
    def __init__(self):
        #SLIP model parameters 

        self.k = 4000 # spring constant 
        self.l_rest = 1 # undeformed leg length
        self.m = 80 # point mass 
        self.g = 9.81

        self.min_leg_torq = -1000
        self.max_leg_torq = 1000

        self.min_spring_const = 4000
        self.max_spring_const = 100000

        self.min_leg_angle = -np.pi/4
        self.max_leg_angle = np.pi/4

        self.collocation_nodes_per_contact = 50

    def stance_feasibility(self, vars):
        """
        Checks if the given state [pos, vel, accel, forcing (inputs) are dynamically feasible]
        Inputs: State
        Outputs: Manipulator Equation Sum (must be zero)
        """
        
        ## Stance Mode Manipulator Equations ##
        #states are in order of x,z,l,theta 
        #x_d, z_d, l_d, theta_d
        # x_dd, z_dd, l_dd, theta_dd
        q = vars[0:4]
        qdot = vars[4:8]
        qddot = vars[8:12]
        u = vars[12:]

        mu_s = 0.74
        mu_k = 0.57

        x = q[0]
        x_d = qdot[0]
        x_dd = qddot[0]
        
        l = q[2]
        l_d = qdot[2]
        l_dd = qddot[2]

        theta = q[3]
        theta_d = qdot[3]
        theta_dd = qddot[3]

        g = self.g
        m = self.m 
        s1 = np.sin(theta)
        c1 = np.cos(theta)
        k = self.k 
        l0 = self.l_rest

        # Coloumb static friction.
        """
        f_s = mu_s*m*l_dd
        f_k = mu_k*m*l_dd
        a = m*(-g*s1 + l*theta_dd + 2*l_d*theta_d)*l        
        
        if f_s < a:
            a = a - u
        else:
            a = a - f_k - u
        """
        #a = 0
        #b = g*m*c1 - k*(l0 - l) + m*l*theta_d**2 - m*l_dd 
        #b = m * g * c1 * l + k*(l0 - l) + 0.5 * m * l * theta_d**2
        #b = 0.5 * 
        a = m * l_dd - m * l * theta_d**2 + m * g * c1 - u[1] * (l0 - l) 
        b = (m * l**2 * theta_dd + 2 * m * l * l_d * theta_d - m * g * l * s1) - u[0]
        return [a,b]
    
    def kinematic_feasibility(self, vars):
        """
        Checks if the given state [pos, vel, accel, forcing (inputs) are kinematically feasible]
        Inputs: State
        Outputs: Manipulator Equation Sum (must be zero)
        """
        
        ## Stance Mode Manipulator Equations ##
        #states are in order of x,z,l,theta 
        #x_d, z_d, l_d, theta_d
        # x_dd, z_dd, l_dd, theta_dd
        q = vars[0:4]
        qdot = vars[4:8]
        qddot = vars[8:12]
        u=vars[12:]
        
        x_d = qdot[0]
        
        l = q[2]
        l_d = qdot[2]
        
        theta = q[3]
        theta_d = qdot[3]
        foot_x_pos = q[0] + np.sin(theta) * l
        a = x_d - np.sin(theta) * l_d - np.cos(theta) * theta_d * l
        #a = x_d + np.sin(theta)*l_d/2 * theta_d*np.cos(theta)*l/2
        #a = 0
        b = 0
        
        return [a,b]
        
    def swing_feasibility(self, qddot):
        """
        Checks if the given swing/flight states are dynamically feasible
        We assume pure ballistic motion during flight, and the leg angle can be instantaneously controlled
        to any angle (leg angle does not show up in the equations of motion)

        Inputs: state variables
        Outputs: Checks if z_double_dot == 0 -g and x_double_dot ==0
        """

        x_dd = qddot[0]
        z_dd = qddot[1]

        equal_zeros = np.array([[x_dd],
                                [z_dd + self.g]])
        
        return [x_dd, z_dd + self.g]

    def get_foot_z_position(self, q):
        """
        foot position is not a state variable, but derived from center of mass
        position and leg configuration 

        useful for checking if the foot is in contact with ground for stance phase constraint
        """

        leg_height = q[2] * np.cos(q[3])
        
        return q[1] - leg_height

    def relate_pelvis_accel_to_leg_accel(self,q,qd,qdd):
        #x,z,l,theta
        ## find accelerations of the pelvis during stance using double derivates of the
        ## kinematic relationship bt x,z and l,theta 
        l1 = q[2]
        theta = q[3]
        l1_dot = qd[2]
        theta_dot = qd[3]
        l1_double_dot = qdd[2]
        theta_double_dot = qdd[3]

        x_dot = -l1*np.cos(theta)*theta_dot - np.sin(theta)*l1_dot
        z_dot = -l1*np.sin(theta)*theta_dot + np.cos(theta)*l1_dot

        x_double_dot = l1*np.sin(theta)*theta_dot**2 \
                        - l1*np.cos(theta)*theta_double_dot \
                        - np.sin(theta)*l1_double_dot \
                        - 2*np.cos(theta)*l1_dot*theta_dot
        
        z_double_dot = -l1*np.sin(theta)*theta_double_dot\
                        - l1*np.cos(theta)*theta_dot**2\
                        - 2*np.sin(theta)*l1_dot*theta_dot\
                        + np.cos(theta)*l1_double_dot

        return [x_double_dot,z_double_dot]

    def create_contact_schedule(self, trajectory_length):
        """
        Given a trajectory length in # of collocation points, find the
        contact schedule as a binary vector of length trajectory length
        
        0 => flight phase
        1 => stance phase

        This is only done once before the optimization 

        ASSUMED: 
        
            The SLIP model begins a trajectory in the beginning
            of a flight phase

            There are 50 collocation points per contact mode

        Input: trajectory length in collocation points

        Output: contact schedule vector 
        """

        if(trajectory_length % self.collocation_nodes_per_contact != 0):
            raise Exception("trajectory length is not divisible by collocation nodes per contact phase")

        contact_schedule = np.zeros(trajectory_length)

        cnpc = self.collocation_nodes_per_contact
        for idx in range(trajectory_length):
            start, end = (idx * cnpc, idx * cnpc + cnpc)
            #if idx is even, then we are in flight phase
            if idx % 2 == 0:
                contact_schedule[start:end] = 0 # 0 means swing
            else:
                contact_schedule[start:end] = 1 # 1 means stance

        return contact_schedule

    def animate(self, q):
        assert len(q.shape) == 2
        assert q.shape[-1] == 4
        
        plt.rcParams["figure.figsize"] = 8,6

        fig, ax = plt.subplots()
        import matplotlib.animation as animation

        ax.axis([-0.2,5,0,3.5])
        
        base, = ax.plot(0, 1, marker="o")
        foot, = ax.plot(0, 1, marker="o")
        line, = ax.plot([], [], color="crimson", zorder=4)

        def update(t):
            t = int(t)
            x, z, l, theta = [q[t,i] for i in range(q.shape[-1])]
            
            base.set_data([x],[z])
            
            foot_z = z - l * np.cos(theta)
            foot_x = x - l * np.sin(theta)
            foot.set_data([foot_x], [foot_z])
            line.set_data([x, foot_x], [z, foot_z])
            #ax.set_xlim(x-1, x+1) #added ax attribute here
            #ax.set_ylim(-0.1, 3) #added ax attribute here

            return base,foot,line

        ani = animation.FuncAnimation(fig, update, interval=100, blit=True, repeat=True,
                            frames=np.linspace(0, len(q), num=len(q), endpoint=False))
        #writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Me'), bitrate=1800)
        #ani.save('traj.mp4', writer=writer)
        plt.show()

def add_linear_constraint(program, constraint, description):
    program.AddLinearConstraint(constraint).evaluator().set_description(description)

def add_nonlinear_constraint(program, constraint, description):
    program.AddConstraint(constraint).evaluator().set_description(description)

def add_bounding_box_constraint(program, variable, min_val, max_val, num_points, description):
    program.AddBoundingBoxConstraint([min_val] * num_points, 
                                     [max_val] * num_points,
                                     variable,
                                     ).evaluator().set_description(description)

def create_trajectory(num_phases: int = 4):
    prog = MathematicalProgram()
    slip_model = SLIP_Model()
    number_of_collocation_points = num_phases * slip_model.collocation_nodes_per_contact 

    q_initial_state = np.array([0, slip_model.l_rest, slip_model.l_rest, 0])
    q_dot_initial_state = np.array([0, 1, 0, 0])

    #q_final_state = np.array([2, 2, slip_model.l_rest, 0])
    #q_dot_final_state = np.array([0, 0, 0, 0])

    # Constraints for max/min timestep lengths in swing and stance
    dt_max = 0.02
    dt_min = 0.002

    min_jump_height = 0.5

    # Decide state positions 
    q = prog.NewContinuousVariables(number_of_collocation_points+1, 4, 'q')

    ground_schedule = slip_model.create_contact_schedule(number_of_collocation_points)

    # Initial position constraint
    add_linear_constraint(prog, q[0,0] == q_initial_state[0], 'x init')
    add_linear_constraint(prog, q[0,1] == q_initial_state[1], 'z init')
    add_linear_constraint(prog, q[0,2] == q_initial_state[2], 'l init')
    add_linear_constraint(prog, q[0,3] == q_initial_state[3], 'theta init')

    # Decide state velocities
    qd = prog.NewContinuousVariables(number_of_collocation_points+1, 4, 'qd')

    # Initial velocity constraint
    add_linear_constraint(prog, qd[0,0] >= q_dot_initial_state[0], 'xd init')
    add_linear_constraint(prog, qd[0,1] == q_dot_initial_state[1], 'zd init')
    add_linear_constraint(prog, qd[0,2] <= q_dot_initial_state[2], 'ld init')
    add_linear_constraint(prog, qd[0,3] <= q_dot_initial_state[3], 'thetad init')

    # Decide state accelerations
    qdd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qdd')

    # Initial acceleration constraint
    add_linear_constraint(prog, qdd[0,2] == 0, 'ldd init')
    add_linear_constraint(prog, qdd[0,3] == 0, 'thetadd init')

    # Decide leg input torques
    u = prog.NewContinuousVariables(number_of_collocation_points, 2, 'u') 

    #Actuator torque bound constraint 
    add_bounding_box_constraint(prog, u[:,0], slip_model.min_leg_torq, slip_model.max_leg_torq,
                                number_of_collocation_points, 'torq bounds')

    # Actuator spring constant constraint
    add_bounding_box_constraint(prog, u[:,1], slip_model.min_spring_const, slip_model.max_spring_const,
                                number_of_collocation_points, 'spring bounds')

    # Actuator leg angle constraint
    add_bounding_box_constraint(prog, q[:,3], slip_model.min_leg_angle, slip_model.max_leg_angle,
                                number_of_collocation_points+1, 'leg angle bounds')

    dts = prog.NewContinuousVariables(num_phases, 1, "dt")

    # Time step constraints 
    add_bounding_box_constraint(prog, dts, dt_min, dt_max, num_phases, 'timestep bound')

    phase_num = 0
    for i in range(number_of_collocation_points-1):
        dt = dts[phase_num]

        if ground_schedule[i] == 0: # Swing
            # Swing dynamics constraints
            for f in slip_model.swing_feasibility(qdd[i,:]):
                add_nonlinear_constraint(prog, f == 0, f'air eqn {i}')

            # Constrain the leg length to be rest length when in swing

            add_linear_constraint(prog, q[i,2] == slip_model.l_rest, f'swing leg length {i}')

            # Euler integration constraint  
            add_linear_constraint(prog, q[i,1] >= min_jump_height, f'min flight body height {i}')
            add_linear_constraint(prog, qdd[i+1,2] == 0, f'no leg dynamics in swing {i}')

            # Foot cannot clip below ground
            add_nonlinear_constraint(prog, slip_model.get_foot_z_position(q[i,:]) >= 0, f'foot above ground {i}')

            prog.AddConstraint(eq(q[i+1,0], q[i,0] + dt * qd[i+1,0])).evaluator().set_description('euler integration flight x pos')
            prog.AddConstraint(eq(qd[i+1,0], qd[i,0] + dt * qdd[i,0])).evaluator().set_description('euler integration flight x vel')
            
            prog.AddConstraint(eq(q[i+1,1], q[i,1] + dt*qd[i+1,1])).evaluator().set_description('euler integration flight z pos')
            prog.AddConstraint(eq(qd[i+1,1], qd[i,1] + dt*qdd[i,1])).evaluator().set_description('euler integration flight z vel')

            # This is so that a single flight phase will work
            if ground_schedule[i] == 1 or i == number_of_collocation_points - 2:
                #add_nonlinear_constraint(prog, slip_model.get_foot_z_position(q[i,:]), None, 1, 'grounded foot')
                #prog.AddConstraint().evaluator().set_description("grounded foot")
                add_nonlinear_constraint
                prog.AddConstraint(slip_model.get_foot_z_position(q[i,:])==0).evaluator().set_description("grounded foot")
                pass
        
        else:
            raise RuntimeError
    # Contact Sequence Dependent Dynamics Constraints 
    """
    for i in range(number_of_collocation_points-1):
        if ground_schedule[i] == 0: # Swing
            # Swing dynamics constraints
            add_nonlinear_constraint(prog, slip_model.swing_feasibility, qdd[i,:], 2, f'air eqn {i}')

            # Constrain the leg length to be rest length when in swing

            add_linear_constraint(prog, q[i,2] == slip_model.l_rest, f'swing leg length {i}')

            # Euler integration constraint  
            prog.AddLinearConstraint(q[i,1]>=min_jump_height).evaluator().set_description("minimum flight body height")
            prog.AddLinearConstraint(qdd[i+1,2]==0).evaluator().set_description('no leg dynamics in swing')
            prog.AddConstraint(slip_model.get_foot_z_position(q[i,:])>=0).evaluator().set_description("foot above ground")

            # prog.AddConstraint(eq(q[i+1,2], q[i,2] + h_swing * qd[i+1,2])).evaluator().set_description('euler integration 1')
            # prog.AddConstraint(eq(qd[i+1,2], qd[i,2] + h_swing * qdd[i,2])).evaluator().set_description('euler integration 2')
            #prog.AddLinearConstraint(qdd[i,2]==0).evaluator().set_description('no leg dynamics in swing')
            # prog.AddLinearConstraint(qdd[i+1,2]==0).evaluator().set_description('no leg dynamics in swing')
            dt = h_swing

            prog.AddConstraint(eq(q[i+1,0], q[i,0] + dt * qd[i+1,0])).evaluator().set_description('euler integration flight x pos')
            prog.AddConstraint(eq(qd[i+1,0], qd[i,0] + dt * qdd[i,0])).evaluator().set_description('euler integration flight x vel')
            
            prog.AddConstraint(eq(q[i+1,1], q[i,1] + dt*qd[i+1,1])).evaluator().set_description('euler integration flight z pos')
            prog.AddConstraint(eq(qd[i+1,1], qd[i,1] + dt*qdd[i,1])).evaluator().set_description('euler integration flight z vel')
            
        else:
            #stance dynamics constraints
            #ensure the foot is on the ground
            var = np.concatenate((q[i,:],qd[i,:],qdd[i,:],u[i,:]))
            #prog.AddConstraint(slip_model.kinematic_feasibility, lb =[0]*2,ub=[0]*2,vars=var).evaluator().set_description("kinematic eqn")

            
            prog.AddLinearConstraint(u[i,1] == u[i-1,1])

            dt = h_stance

            #euler integration constraint  
            if ground_schedule[i+1]==0:
              #next step is swing 
              #only integrate body x,z positions and vels 
              prog.AddConstraint(eq(q[i+1,0], q[i,0] + dt * qd[i+1,0])).evaluator().set_description('stance euler integration x pos')
              prog.AddConstraint(eq(qd[i+1,0], qd[i,0] + dt * qdd[i,0])).evaluator().set_description('stance euler integration z pos')
              
              prog.AddConstraint(eq(q[i+1,1], q[i,1] + dt * qd[i+1,1])).evaluator().set_description('stance euler integration x vel')
              prog.AddConstraint(eq(qd[i+1,1], qd[i,1] + dt * qdd[i,1])).evaluator().set_description('stance euler integration z pos')
              #prog.AddLinearConstraint(qdd[i,2]==0).evaluator().set_description('no leg dynamics in swing')

            else:

              prog.AddConstraint(eq(qd[i+1,0], qd[i,0] + dt * qdd[i,0])).evaluator().set_description('stance euler integration z pos')
              prog.AddConstraint(eq(qd[i+1,1], qd[i,1] + dt * qdd[i,1])).evaluator().set_description('stance euler integration z pos')


              #prog.AddConstraint(eq(q[i+1,:], q[i,:] + dt * qd[i+1,:])).evaluator().set_description('stance full state euler integration pos')
              #prog.AddConstraint(eq(qd[i+1,:], qd[i,:] + dt * qdd[i,:])).evaluator().set_description('stance full state euler integration vel')

              prog.AddConstraint(eq(q[i+1,2], q[i,2] + dt * qd[i+1,2])).evaluator().set_description('stance full state euler integration pos')
              prog.AddConstraint(eq(qd[i+1,2], qd[i,2] + dt * qdd[i,2])).evaluator().set_description('stance full state euler integration vel')
              
              prog.AddConstraint(eq(q[i+1,3], q[i,3] + dt * qd[i+1,3])).evaluator().set_description('stance full state euler integration pos')
              prog.AddConstraint(eq(qd[i+1,3], qd[i,3] + dt * qdd[i,3])).evaluator().set_description('stance full state euler integration vel')
        
              prog.AddConstraint(slip_model.kinematic_feasibility, lb =[0]*2,ub=[0]*2,vars=var).evaluator().set_description("kinematic eqn")
              prog.AddConstraint(qdd[i,0]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[0])
              prog.AddConstraint(qdd[i,1]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[1])
        
              prog.AddConstraint(slip_model.get_foot_z_position(q[i,:])==0).evaluator().set_description("grounded foot")
              prog.AddLinearConstraint(q[i,2]>=0.5).evaluator().set_description("min stance height")
              prog.AddConstraint(slip_model.stance_feasibility, lb=[0]*2,ub=[0]*2,vars=var).evaluator().set_description("stance eqn")
        prog.AddCost(dt[0]*u[i].dot(u[i]))
    """

    solver = SnoptSolver()
    result = solver.Solve(prog)

    pos = result.GetSolution(q)
    vel = result.GetSolution(qd)
    accel = result.GetSolution(qdd)
    inputs = result.GetSolution(u)

    if not result.is_success():
        for constraint in result.GetInfeasibleConstraintNames(prog):
            print(constraint)

        raise RuntimeError("Could not generate trajectory")

    slip_model.animate(pos)
create_trajectory(1)
