import sys
sys.path.append('./underactuated')
import numpy as np 
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, eq, SnoptSolver
from underactuated import running_as_notebook
from underactuated.jupyter import AdvanceToAndVisualize
import os

class SLIP_Model():
    def __init__(self):
        #SLIP model parameters 

        self.k = 4000 # spring constant 
        self.l_rest = 1 # undeformed leg length
        self.m = 80 # point mass 
        self.g = 9.81

        self.min_spring_const = 500
        self.max_spring_const = 15000

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

        a = x_d - np.sin(theta) * l_d - np.cos(theta) * theta_d * l
        #a = x_d - np.sin(theta) * l_d - np.cos(theta) * theta_d * l

        #return [a]

        #foot_x_pos = q[0] + np.sin(theta) * l
        #a = x_d - np.sin(theta) * l_d - np.cos(theta) * theta_d * l
        #b = z_d + np.cos(theta) * l_d - np.sin(theta) * theta_d * l
        #a = x_d + np.sin(theta)*l_d/2 * theta_d*np.cos(theta)*l/2
        #a = 0
        #b = np.sqrt(np.sum(q[:2] - [foot_x_pos, q[1] + np.cos(theta) * l])**2)
        #b = 0
        
        return [a]
        
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

        return [x_dot,z_dot,x_double_dot,z_double_dot]


    def energy(self, q, qd, u):
        T = self.m/2 * (qd[2] ** 2 + q[2] * qd[3]**2)
        U = self.m * self.g * q[2] * np.cos(q[3]) + u/2 * (self.l_rest - q[2])**2
        return T + U

    def animate(self, q, save_to=None):
        assert len(q.shape) == 2
        assert q.shape[-1] == 4
        
        plt.rcParams["figure.figsize"] = 8,6

        fig, ax = plt.subplots()
        import matplotlib.animation as animation

        ax.axis([-2,2,0,2])
        
        base, = ax.plot(0, 1, marker="o", markersize=20)
        foot, = ax.plot(0, 1, marker="o", markersize=10)
        line, = ax.plot([], [], color="crimson", zorder=4, markersize=5)

        def update(t):
            t = int(t)
            x, z, l, theta = [q[t,i] for i in range(q.shape[-1])]
            
            base.set_data([x],[z])
            
            foot_z = z - l * np.cos(theta)
            foot_x = x - l * np.sin(theta)
            foot.set_data([foot_x], [foot_z])
            line.set_data([x, foot_x], [z, foot_z])

            return base,foot,line

        ani = animation.FuncAnimation(fig, update, interval=30, blit=True, repeat=True,
                            frames=np.linspace(0, len(q), num=len(q), endpoint=False))
        if save_to is not None:
            writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(save_to, writer=writer)
            print(save_to)
            os.system(f'ffmpeg -i {save_to} -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 "{save_to[:-4]}.gif"')
        else:
            plt.show()

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

def create_swing_trajectory(
        slip_model,
        initial_x_vel = None,
        initial_z_vel = None,
        initial_x = None,
        initial_z = None,
        initial_theta = None,
        final_x_vel = None,
        final_z_vel = None,
        final_x = None,
        final_z = None,
        final_theta = None,
):
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
    if initial_x is not None:
        add_linear_constraint(prog, q[0,0] == initial_x, 'x init')

    if initial_z is not None:
        add_linear_constraint(prog, q[0,1] == initial_z, 'z init')

    if initial_theta is not None:
        add_linear_constraint(prog, q[0,3] == initial_theta, 'theta init')

    if final_x is not None:
        add_linear_constraint(prog, q[-1,0] == final_x, 'x final')

    if final_z is not None:
        add_linear_constraint(prog, q[-1,1] == final_z, 'z final')

    if final_theta is not None:
        add_linear_constraint(prog, q[-1,3] == final_theta, 'theta final')

    # Decide state velocities
    qd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qd')

    # Initial velocity constraint
    if initial_x_vel is not None:
        add_linear_constraint(prog, qd[0,0] == initial_x_vel, 'xd init')

    if initial_z_vel is not None:
        add_linear_constraint(prog, qd[0,1] == initial_z_vel, 'zd init')

    if final_x_vel is not None:
        add_linear_constraint(prog, qd[-1,0] == final_x_vel, 'xd final')

    if final_z_vel is not None:
        add_linear_constraint(prog, qd[-1,1] == final_z_vel, 'zd final')

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
    prog.AddConstraint(slip_model.angle_matches_velocity, lb=[-0.5], ub=[0.5], vars=var).evaluator().set_description("angle match")

    var = np.concatenate((q[-1,:],qd[-1,:]))
    #prog.AddConstraint(slip_model.angle_matches_velocity, lb=[-0.5], ub=[0.5], vars=var).evaluator().set_description("angle match")


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

        var = np.concatenate((q[i,:],qd[i,:]))
        #prog.AddConstraint(slip_model.angle_matches_velocity, lb=[-0.5], ub=[0.5], vars=var).evaluator().set_description("angle match")

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

    if not result.is_success():
        for constraint in result.GetInfeasibleConstraintNames(prog)[:4]:
            print(constraint)

        slip_model.animate(pos)
        raise RuntimeError("Could not generate trajectory")
    return pos, vel

def create_stance_trajectory(
        slip_model,
        initial_x_vel = None,
        initial_z_vel = None,
        initial_x = None,
        initial_z = None,
        initial_theta = None,
        final_x_vel = None,
        final_z_vel = None,
        final_x = None,
        final_z = None,
        final_theta = None,
):
    prog = MathematicalProgram()
    number_of_collocation_points = slip_model.collocation_nodes_per_contact 

    #constraints for max/min timestep lengths in swing and stance

    max_dt = 0.035
    min_dt = 0.01

    dt = prog.NewContinuousVariables(1,"dt")

    #decide state positions 
    q = prog.NewContinuousVariables(number_of_collocation_points, 4, 'q')
    #decide state velocities
    qd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qd')
    #decide state accelerations
    qdd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qdd')
    #decide leg input torques
    u = prog.NewContinuousVariables(1, 1, 'u') 

    # Initial position constraint
    if initial_x is not None:
        add_linear_constraint(prog, q[0,0] == initial_x, 'x init')

    if initial_z is not None:
        add_linear_constraint(prog, q[0,1] == initial_z, 'z init')

    prog.AddLinearConstraint(q[0,2] == slip_model.l_rest)

    if initial_theta is not None:
        add_linear_constraint(prog, q[0,3] == initial_theta, 'theta init')

    if final_x is not None:
        add_linear_constraint(prog, q[-1,0] == final_x, 'x final')

    if final_z is not None:
        add_linear_constraint(prog, q[-1,1] == final_z, 'z final')

    if final_theta is not None:
        add_linear_constraint(prog, q[-1,3] == final_theta, 'theta final')

    # Initial velocity constraint
    if initial_x_vel is not None:
        add_linear_constraint(prog, qd[0,0] == initial_x_vel, 'xd init')

    if initial_z_vel is not None:
        add_linear_constraint(prog, qd[0,1] == initial_z_vel, 'zd init')
    else:
        add_linear_constraint(prog, qd[0,1] <= 0, 'zd init')

    if final_x_vel is not None:
        add_linear_constraint(prog, qd[-1,0] == final_x_vel, 'xd final')

    if final_z_vel is not None:
        add_linear_constraint(prog, qd[-1,1] == final_z_vel, 'zd final')
    else:
        add_linear_constraint(prog, qd[-1,1] >= 0, 'zd final')

    #Actuator spring constant constraint
    prog.AddBoundingBoxConstraint([slip_model.min_spring_const]*1, 
                                  [slip_model.max_spring_const]*1,
                                  u[:,0]
                                  ).evaluator().set_description('sprint constant constraint')
    #Actuator leg angle constraint 
    prog.AddBoundingBoxConstraint([slip_model.min_leg_angle]*(number_of_collocation_points), 
                                  [slip_model.max_leg_angle]*(number_of_collocation_points), q[:,3]
                                  ).evaluator().set_description('leg_angle_bounds')

    prog.AddBoundingBoxConstraint(min_dt,max_dt,dt).evaluator().set_description('stance time bound')

    # Contact Sequence Dependent Dynamics Constraints 
    for i in range(number_of_collocation_points):
        #stance dynamics constraints
        #ensure the foot is on the ground
        var = np.concatenate((q[i,:],qd[i,:],qdd[i,:],u[0,:]))

        prog.AddLinearConstraint(q[i,2] <= slip_model.l_rest)
        prog.AddConstraint(slip_model.kinematic_feasibility, lb =[0],ub=[0],vars=var).evaluator().set_description("kinematic eqn")
        prog.AddConstraint(slip_model.get_foot_z_position(q[i]) == 0).evaluator().set_description("grounded foot")
        prog.AddLinearConstraint(q[i,1]>=0.2).evaluator().set_description("min stance height")
        prog.AddConstraint(slip_model.stance_feasibility, lb=[0]*2,ub=[0]*2,vars=var).evaluator().set_description("stance eqn")

        if i < number_of_collocation_points - 1:
            prog.AddConstraint(eq(q[i+1,:], q[i,:] + dt * qd[i,:])).evaluator().set_description('stance full state euler integration pos')
            prog.AddConstraint(eq(qd[i+1,:], qd[i,:] + dt * qdd[i,:])).evaluator().set_description('stance full state euler integration vel')

        #if i == number_of_collocation_points - 1:
    prog.AddConstraint(qd[-1,0]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[0])
    prog.AddConstraint(qd[-1,1]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[1])
    prog.AddConstraint(qdd[-1,0]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[2])
    prog.AddConstraint(qdd[-1,1]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[3])
    prog.AddLinearConstraint(q[-1,2] == slip_model.l_rest)
       
    solver = SnoptSolver()
    result = solver.Solve(prog)

    pos = result.GetSolution(q)
    vel = result.GetSolution(qd)
    accel = result.GetSolution(qdd)
    inputs = result.GetSolution(u)

    if not result.is_success():
        #print(result.GetInfeasibleConstraintNames(prog))
        print(inputs)
        print(result.GetSolution(dt))

        raise RuntimeError("Could not generate trajectory")

    #slip_model.animate(pos)
    return pos, vel

if __name__ == '__main__':
    m = SLIP_Model()
    takeoff_velocity = 3
    takeoff_angle = np.pi/2 - 1e-1

    # We want to reach some apex height
    target_height = 2
    takeoff_yvel = np.sqrt((target_height - m.l_rest) * 2 * m.g)

    #pos, vel = create_stance_trajectory(m, final_x_vel=1, final_z_vel=5)
    #m.animate(pos)
    #successes = np.zeros((10,10))
    #successes = np.zeros((10))
    #for i, x in enumerate(np.linspace(-3.5, 3.5, num=10)):
    #    #for j, z in enumerate(np.linspace(-1, -4, num=10)):
    #            x_f = 1
    #            try:
    #                pos, vel = create_stance_trajectory(m, final_x_vel=x)#, final_z_vel=z)
    #                successes[i][j] = 1
    #                #print("success", x, z)
    #                print('success', x)
    #                #m.animate(pos)
    #            except RuntimeError:
    #                #print("fail", x, z)
    #                print('fail', x)
    #                pass
    #print(successes)

    #exit(0)
    #for x_i in [0, 1, 2]:
    #    traj, vel = create_swing_trajectory(m, initial_x_vel=x_i, initial_z_vel=4)
    #    m.animate(traj, save_to=f'swing{x_i}.mp4')
    #    print("did ", x_i)
    #exit(0)

    if True:
        # This always ends up doing bouncing with zero x velocity?
        traj, vel = create_swing_trajectory(m, final_theta=-np.pi/6, initial_x_vel=3, initial_z_vel=2.5)
        try:
            print("touchdown vels:", vel[-1,:2])
            pos, vel = create_stance_trajectory(m, initial_x_vel=vel[-1,0], initial_z_vel=vel[-1,1])
            pos[:,0] = pos[:,0] + traj[-1,0]
            traj = np.vstack([traj, pos])
            pos, vel = create_swing_trajectory(m, initial_x_vel=vel[-1,0], initial_z_vel=vel[-1,1])
            pos[:,0] = pos[:,0] + traj[-1,0]
            traj = np.vstack([traj, pos])
            pos, vel = create_stance_trajectory(m, initial_x_vel=vel[-1,0], initial_z_vel=vel[-1,1])
            pos[:,0] = pos[:,0] + traj[-1,0]
            traj = np.vstack([traj, pos])
            pos, vel = create_swing_trajectory(m, initial_x_vel=vel[-1,0], initial_z_vel=vel[-1,1])
            pos[:,0] = pos[:,0] + traj[-1,0]
            traj = np.vstack([traj, pos])
            pos, vel = create_stance_trajectory(m, initial_x_vel=vel[-1,0], initial_z_vel=vel[-1,1])
            pos[:,0] = pos[:,0] + traj[-1,0]
            traj = np.vstack([traj, pos])
        except RuntimeError:
            import traceback
            traceback.print_exc()
    if False: 
        pos, vel = create_swing_trajectory(m, initial_x_vel=0, initial_z_vel=3)
        traj = pos

        try:
            # Use the initial conditions of the swing to create a stance trajectory
            print("z vel targets1: ", vel[0][:2])
            pos, vel = create_stance_trajectory(m, initial_x_vel=1, final_x_vel=vel[0][0], final_z_vel=vel[0][1])
            pos[:,0] = pos[:,0] - pos[-1,0] + traj[0,0]
            traj = np.vstack([pos, traj])

            # Use the initial conditions of THAT stance trajectory to create the preceding swing trajectory
            print("z vel targets2: ", vel[0][:2])
            pos, vel = create_swing_trajectory(m, final_theta=pos[0][3], final_x_vel=vel[0][0], final_z_vel=vel[0][1])
            pos[:,0] = pos[:,0] - pos[-1,0] + traj[0,0]
            traj = np.vstack([pos, traj])

            # Use the initial conditions of the swing to create a stance trajectory
            #print("z vel targets3: ", vel[0][:2])
            #pos, vel = create_stance_trajectory(m, initial_x_vel=np.random.uniform(-1, 1), final_x_vel=vel[0][0], final_z_vel=vel[0][1])
            #pos[:,0] = pos[:,0] - pos[-1,0] + traj[0,0]
            #traj = np.vstack([pos, traj])

            # Use the initial conditions of THAT stance trajectory to create the preceding swing trajectory
            #pos, vel = create_swing_trajectory(m, final_theta=pos[0][3], final_x_vel=vel[0][0], final_z_vel=vel[0][1])
            #pos[:,0] = pos[:,0] - pos[-1,0] + traj[0,0]
            #traj = np.vstack([pos, traj])

        except RuntimeError:
            import traceback
            traceback.print_exc()
    m.animate(traj, save_to='enchilada2.mp4')

