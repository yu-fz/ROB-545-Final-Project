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

        self.min_spring_const = 100
        self.max_spring_const = 10000

        self.min_leg_angle = -np.pi/3
        self.max_leg_angle = np.pi/3

        self.collocation_nodes_per_contact = 40

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

        a = m * l_dd - m * l * theta_d**2 + m * g * c1 - u[0] * (l0 - l) 
        b = (m * l**2 * theta_dd + 2 * m * l * l_d * theta_d - m * g * l * s1)

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

        x = q[0]
        x_d = qdot[0]
        x_dd = qddot[0]
        
        l = q[2]
        l_d = qdot[2]
        l_dd = qddot[2]
        
        x_d = qdot[0]
        z_d = qdot[1]
        
        theta = q[3]
        theta_d = qdot[3]
        theta_ddot = qddot[3]

        # Foot cartesian acceleration should be zero
        xdd_foot = -l * np.sin(theta) * theta_ddot**2 +\
            l * np.cos(theta) * theta_ddot**2 +\
            np.sin(theta) * l_dd +\
            2 * np.cos(theta) * l_d * theta_d

        zdd_foot = l * np.sin(theta) * theta_ddot**2 +\
            l * np.cos(theta) * theta_ddot**2 +\
            2 * np.sin(theta) * l_d * theta_ddot -\
            np.cos(theta) * l_dd + self.g

        xd_foot = qdot[0] - l * np.cos(theta) * theta_d - np.sin(theta) * l_d
        zd_foot = qdot[1] + l * np.sin(theta) * theta_d - np.cos(theta) * l_d
        #a = x_d - np.sin(theta) * l_d - np.cos(theta) * theta_d * l

        #return [a]

        #foot_x_pos = q[0] + np.sin(theta) * l
        #a = x_d - np.sin(theta) * l_d - np.cos(theta) * theta_d * l
        #b = z_d + np.cos(theta) * l_d - np.sin(theta) * theta_d * l
        #a = x_d + np.sin(theta)*l_d/2 * theta_d*np.cos(theta)*l/2
        #a = 0
        #b = np.sqrt(np.sum(q[:2] - [foot_x_pos, q[1] + np.cos(theta) * l])**2)
        #b = 0
        
        return [xd_foot, 0, 0, 0]
        
    def angle_matches_velocity(self, var):
        q = var[:4]
        qd = var[4:]
        angle = q[3]
        if qd[0] != 0.0 and qd[1] != 0.0:
            vel_angle = np.arctan2(qd[0], qd[1])
        else:
            vel_angle = 0
        return [angle - vel_angle]

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

        #q = vars[:4]
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

        return [x_double_dot, z_double_dot]


    def energy(self, q, qd, u):
        #k = 0.5 * u[1] * (q[2] - self.l_rest)**2
        #v = self.m/2 * (qd[0] ** 2 + qd[1] ** 2)
        #p = self.m * self.g * q[1]
        T = self.m/2 * (qd[2] ** 2 + q[2] * qd[3]**2)
        U = self.m * self.g * q[2] * np.cos(q[3]) + u/2 * (self.l_rest - q[2])**2
        return T + U

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

        ax.axis([-5,5,0,5])
        
        base, = ax.plot(0, 1, marker="o")
        foot, = ax.plot(0, 1, marker="o")
        line, = ax.plot([], [], color="crimson", zorder=4)

        def update(t):
            t = int(t)
            x, z, l, theta = [q[t,i] for i in range(q.shape[-1])]
            
            base.set_data([x],[z])
            
            foot_z = z - l * np.cos(theta)
            foot_x = x + l * np.sin(theta)
            foot.set_data([foot_x], [foot_z])
            line.set_data([x, foot_x], [z, foot_z])
            #ax.set_xlim(x-1, x+1) #added ax attribute here
            #ax.set_ylim(0, 3) #added ax attribute here

            return base,foot,line

        ani = animation.FuncAnimation(fig, update, interval=10, blit=True, repeat=True,
                            frames=np.linspace(0, len(q), num=len(q), endpoint=False))
        writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('traj.mp4', writer=writer)
        plt.show()
    def wtf(self, vars):
        q = vars[:4]
        q_next = vars[4:]

        equal = (q[0] + q[2] * np.sin(q[3])) - (q_next[0] + q_next[2] * np.sin(q_next[3]))
        return [equal]

def add_linear_constraint(program, constraint, description):
    program.AddLinearConstraint(constraint).evaluator().set_description(description)

def add_nonlinear_constraint(program, constraint, description):
    program.AddConstraint(constraint).evaluator().set_description(description)

def add_bounding_box_constraint(program, variable, min_val, max_val, num_points, description):
    if num_points > 1:
        program.AddBoundingBoxConstraint([min_val] * num_points, 
                                         [max_val] * num_points,
                                         variable,
                                         ).evaluator().set_description(description)
    else:
        program.AddBoundingBoxConstraint(min_val, 
                                         max_val,
                                         variable,
                                         ).evaluator().set_description(description)

def create_swing_trajectory(slip_model, q_i = None, qd_i = None, q_f = None, qd_f = None):
    prog = MathematicalProgram()
    #slip_model = SLIP_Model()
    number_of_collocation_points = slip_model.collocation_nodes_per_contact 

    dt_max = 0.05
    dt_min = 0.005

    min_jump_height = 0.5

    # Decide state positions 
    q = prog.NewContinuousVariables(number_of_collocation_points, 4, 'q')

    #assert None in [q_i, q_f] and None in [qd_i, qd_f]

    # Initial position constraint
    if q_i is not None:
        assert len(q_i) == 2
        add_linear_constraint(prog, q[0,0] == q_i[0], 'x init')
        add_linear_constraint(prog, q[0,1] == q_i[1], 'z init')

    if q_f is not None:
        assert len(q_f) == 2
        add_linear_constraint(prog, q[-1,0] == q_f[0], 'x final')
        add_linear_constraint(prog, q[-1,1] == q_f[1], 'z final')

    # Decide state velocities
    qd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qd')

    # Initial velocity constraint
    if qd_i is not None:
        add_linear_constraint(prog, qd[0,0] == qd_i[0], 'xd init')
        add_linear_constraint(prog, qd[0,1] == qd_i[1], 'zd init')

    if qd_f is not None:
        add_linear_constraint(prog, qd[-1,0] == qd_f[0], 'xd final')
        add_linear_constraint(prog, qd[-1,1] == qd_f[1], 'zd final')

    # Decide state accelerations
    qdd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qdd')

    # Initial acceleration constraint
    add_linear_constraint(prog, qdd[0,2] == 0, 'ldd init')
    add_linear_constraint(prog, qdd[0,3] == 0, 'thetadd init')

    # Actuator leg angle constraint
    add_bounding_box_constraint(prog, q[:,3], slip_model.min_leg_angle, slip_model.max_leg_angle,
                                number_of_collocation_points, 'leg angle bounds')

    dt = prog.NewContinuousVariables(1, 1, "dt")

    # Time step constraints 
    add_bounding_box_constraint(prog, dt, dt_min, dt_max, 1, 'timestep bound')

    var = np.concatenate((q[0,:],qd[0,:]))
    prog.AddConstraint(slip_model.angle_matches_velocity, lb=[0], ub=[0], vars=var).evaluator().set_description("angle match")

    var = np.concatenate((q[-1,:], qd[-1,:]))
    #prog.AddConstraint(slip_model.angle_matches_velocity, lb=[0], ub=[0], vars=var).evaluator().set_description("angle match")

    # Foot must touch ground at end of swing
    add_nonlinear_constraint(prog, slip_model.get_foot_z_position(q[0,:]) == 0, f'grounded foot {0}')
    add_nonlinear_constraint(prog, slip_model.get_foot_z_position(q[-1,:]) == 0, f'grounded foot {-1}')

    for i in range(number_of_collocation_points):
        #add_nonlinear_constraint(prog, slip_model.energy(q[i,:], qd[i,:], u[phase_num,:]) == slip_model.energy(q[i+1,:], qd[i+1,:], u[i+1,:]), f'energy constraint {i} ')

        # Swing dynamics constraints
        for f in slip_model.swing_feasibility(qdd[i,:]):
            add_nonlinear_constraint(prog, f == 0, f'air eqn {i}')

        # Constrain the leg length to be rest length when in swing

        add_linear_constraint(prog, q[i,2] == slip_model.l_rest, f'swing leg length {i}')

        #add_linear_constraint(prog, q[i,1] >= min_jump_height, f'min flight body height {i}')
        add_linear_constraint(prog, qdd[i,2] == 0, f'no leg dynamics in swing {i}')

        # Euler integration constraint
        if i < number_of_collocation_points - 1:
            add_nonlinear_constraint(prog, eq(q[i+1,:], q[i,:] + dt * qd[i+1,:]), f'euler integration flight x vel {i}')
            add_nonlinear_constraint(prog, eq(qd[i+1,:], qd[i,:] + dt * qdd[i,:]), f'euler integration flight x vel {i}')

            # Foot cannot clip below ground
            if i > 0:
                add_nonlinear_constraint(prog, slip_model.get_foot_z_position(q[i,:]) >= 0, f'foot above ground {i}')
        
    solver = SnoptSolver()
    result = solver.Solve(prog)

    pos = result.GetSolution(q)
    vel = result.GetSolution(qd)
    accel = result.GetSolution(qdd)
    dt = result.GetSolution(dt)
    print(pos[:,2])
    print(dt)

    if not result.is_success():
        for constraint in result.GetInfeasibleConstraintNames(prog)[:4]:
            print(constraint)

        slip_model.animate(pos)
        raise RuntimeError("Could not generate trajectory")
    plt.plot(pos[:-1,0],label = "x distance")
    plt.plot(pos[:-1,1],label = "z height")
    plt.plot(pos[:-1,2],label = "leg length")
    #plt.plot(ts, pos[:-1,3],label = "leg angle")
    plt.legend()
    plt.show()

    plt.plot(vel[:,0][:-1],label = "x vel")
    plt.plot(vel[:,1][:-1],label = "z vel")
    plt.legend()
    plt.show()
    slip_model.animate(pos)
    return pos, vel, accel, dt

def create_stance_trajectory(slip_model, qd_i = None, exit_angle = None):
    prog = MathematicalProgram()
    number_of_collocation_points = slip_model.collocation_nodes_per_contact 

    # Constraints for max/min timestep lengths in swing and stance
    dt_max = 0.01
    dt_min = 0.000005

    # Decide state positions 
    q = prog.NewContinuousVariables(number_of_collocation_points, 4, 'q')

    # Initial position constraint
    #add_linear_constraint(prog, q[0,0] == q_initial_state[0], 'x init')
    #add_linear_constraint(prog, q[0,1] == q_initial_state[1], 'z init')
    #add_linear_constraint(prog, q[0,3] == q_initial_state[3], 'theta init')

    # Decide state velocities
    qd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qd')

    # Initial velocity constraint
    if qd_i is not None:
        add_linear_constraint(prog, qd[0,0] == qd_i[0], 'xd init')
        add_linear_constraint(prog, qd[0,1] == qd_i[1], 'zd init')
    #add_linear_constraint(prog, qd[-1,0] == -qd_i[0], 'zd final')
    add_linear_constraint(prog, qd[-1,1] == qd_i[0], 'zd final')
    #add_linear_constraint(prog, qd[0,2] == q_dot_initial_state[2], 'ld init')
    #add_linear_constraint(prog, qd[0,3] == q_dot_initial_state[3], 'thetad init')

    # Decide state accelerations
    qdd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qdd')

    # Initial acceleration constraint
    #add_linear_constraint(prog, qdd[0,2] == 0, 'ldd init')
    #add_linear_constraint(prog, qdd[0,3] == 0, 'thetadd init')

    # Decide leg spring stiffness
    u = prog.NewContinuousVariables(1, 1, 'u') 

    # Actuator spring constant constraint
    add_bounding_box_constraint(prog, u[:,0], slip_model.min_spring_const, slip_model.max_spring_const,
                                1, 'spring bounds')

    dt = prog.NewContinuousVariables(1, 1, "dt")

    # Time step constraints 
    add_bounding_box_constraint(prog, dt, dt_min, dt_max, 1, 'timestep bound')

    # Actuator leg angle constraint
    add_bounding_box_constraint(prog, q[:,3], slip_model.min_leg_angle, slip_model.max_leg_angle,
                                number_of_collocation_points, 'leg angle bounds')

    add_linear_constraint(prog, q[0,2] == slip_model.l_rest, f'stance leg length {0}')
    for i in range(number_of_collocation_points):
        var = np.concatenate((q[i,:], qd[i,:], qdd[i,:], u[0,:]))
        prog.AddConstraint(slip_model.stance_feasibility, lb=[0]*2,ub=[0]*2,vars=var).evaluator().set_description("stance eqn")
        #prog.AddConstraint(slip_model.kinematic_feasibility, lb = [0]*4, ub = [0]*4, vars=var).evaluator().set_description("kinematic eqn")
        #add_nonlinear_constraint(prog, slip_model.energy(q[i,:], qd[i,:], u[phase_num,:]) == slip_model.energy(q[i+1,:], qd[i+1,:], u[i+1,:]), f'energy constraint {i} ')

        if i < number_of_collocation_points - 1:
            prog.AddConstraint(eq(q[i+1,:], q[i,:] + dt * qd[i,:])).evaluator().set_description(f'stance full state euler integration pos {i}')
            prog.AddConstraint(eq(qd[i+1,:], qd[i,:] + dt * qdd[i,:])).evaluator().set_description(f'stance full state euler integration vel {i}')

            add_linear_constraint(prog, q[i,2] <= slip_model.l_rest, f'stance leg length {i}')

            var = np.concatenate((q[i,:], q[i+1,:]))
            #prog.AddConstraint(slip_model.wtf, lb=[0], ub=[0], vars=var, description='foot cannot move')
        #prog.AddConstraint(qdd[i,0]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[0])
        #prog.AddConstraint(qdd[i,1]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[1])

        #prog.AddLinearConstraint(q[i,2]>=0.5).evaluator().set_description("min stance height")
        prog.AddConstraint(slip_model.get_foot_z_position(q[i,:])==0).evaluator().set_description("grounded foot")

    add_linear_constraint(prog, q[-1,2] == slip_model.l_rest, f'stance leg length {-1}')
    add_linear_constraint(prog, q[-1,3] == 0, f'theta end {-1}')
    add_linear_constraint(prog, qd[-1,1] >= 0, 'nonnegative liftoff z velocity')

    solver = SnoptSolver()
    result = solver.Solve(prog)

    pos = result.GetSolution(q)
    vel = result.GetSolution(qd)
    accel = result.GetSolution(qdd)
    inputs = result.GetSolution(u)
    dt = result.GetSolution(dt)
    #energy = np.array([slip_model.energy(pos[i], vel[i], inputs[i]) for i in range(len(pos)-1)])
    #print(dts)
    print(dt)
    print(inputs)
    #ts = np.cumsum(dts)

    if not result.is_success():
        for constraint in result.GetInfeasibleConstraintNames(prog)[:4]:
            print(constraint)

        print("Total constraints failed:", len(result.GetInfeasibleConstraintNames(prog)))
        slip_model.animate(pos)
        #raise RuntimeError("Could not generate trajectory")

    print(pos[:,0])
    foot_acc = np.array([slip_model.kinematic_feasibility(np.hstack([pos[i], vel[i], accel[i], inputs])) for i in range(len(pos))])
    #plt.plot(foot_acc[:,0], label='foot x acc')
    #plt.plot(foot_acc[:,1], label='foot z acc')
    #plt.plot(foot_acc[:,2], label='foot x vel')
    #plt.plot((pos[:,0] + pos[:,2] * np.sin(pos[:,3])), label='foot x pos')
    #plt.plot((pos[:,0] + pos[:,2] * np.sin(pos[:,3]))[1:] - (pos[:,0] + pos[:,2] * np.sin(pos[:,3]))[:-1], label='foot x delta')
    #plt.plot(foot_acc[:,3], label='foot z vel')
    #plt.legend()
    #plt.show()

    #plt.plot(pos[:-1,0],label = "x distance")
    #plt.plot(pos[:-1,1],label = "z height")
    plt.plot(pos[:-1,2],label = "leg length")
    #plt.plot(pos[:-1,3],label = "leg angle")
    #plt.plot(vel[:,0][:-1],label = "x vel")
    #plt.plot(vel[:,1][:-1],label = "z vel")
    plt.legend()
    plt.show()

    #plt.plot(energy, label = "energy")
    #plt.legend()
    #plt.show()


    plt.plot(vel[:,0][:-1],label = "x vel")
    plt.plot(vel[:,1][:-1],label = "z vel")
    plt.legend()
    plt.show()
    # plt.plot(accel[:,0])
    # plt.plot(accel[:,1])
    #plt.plot(accel[:,1])

    #plt.show()
    #plt.plot(inputs[:,0], label = 'input 0')
    #plt.plot(inputs[:,1], label = 'input 1')
    #plt.legend()
    #plt.show()
    #slip_model.animate(pos)

if __name__ == '__main__':
    m = SLIP_Model()
    takeoff_velocity = 3
    takeoff_angle = np.pi/2 - 1e-1
    #q_i = np.array([np.cos(takeoff_angle) * m.l_rest, np.sin(takeoff_angle) * m.l_rest])
    #qd_i = np.array([np.cos(takeoff_angle) * takeoff_velocity, np.sin(takeoff_angle) * takeoff_velocity])
    #qd_i = np.array([1, np.sin(takeoff_angle) * takeoff_velocity])
    if False:
        qd_i = [2, 4]
        create_swing_trajectory(m, qd_i=qd_i)
    else:
        qd_i = [3, -6]
        create_stance_trajectory(m, qd_i=qd_i)
