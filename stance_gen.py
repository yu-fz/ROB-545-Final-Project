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

        self.min_spring_const = 2000
        self.max_spring_const = 15000

        self.min_leg_angle = -np.pi/3
        self.max_leg_angle = np.pi/3

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
        
        x_d = qdot[0]
        
        l = q[2]
        l_d = qdot[2]
        
        theta = q[3]
        theta_d = qdot[3]
        foot_x_pos = q[0] + np.sin(theta) * l
        a = x_d - np.sin(theta) * l_d - np.cos(theta) * theta_d * l
        
        return [a]
        
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
        
        return [x_dd,z_dd + self.g]

    def get_foot_z_position(self, vars):
        """
        foot position is not a state variable, but derived from center of mass
        position and leg configuration 

        useful for checking if the foot is in contact with ground for stance phase constraint
        """

        q = vars[0:4]

        leg_height = q[2] * np.cos(q[3])
        
        return [q[1] - leg_height]

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
                        + np.cos(theta)*l1_double_dot - self.g

        return [x_dot,z_dot,x_double_dot,z_double_dot]

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

        ani = animation.FuncAnimation(fig, update, interval=30, blit=True, repeat=True,
                            frames=np.linspace(0, len(q), num=len(q), endpoint=False))
        writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('traj.mp4', writer=writer)
        plt.show()

prog = MathematicalProgram()
slip_model = SLIP_Model()
number_of_collocation_points = 1 * slip_model.collocation_nodes_per_contact 

q_initial_state = np.array([0,1,slip_model.l_rest,0])
q_dot_initial_state = np.array([2,0,0,0])

q_final_state = np.array([0,0,slip_model.l_rest,0])
q_dot_final_state = np.array([1,0,0,0])

#constraints for max/min timestep lengths in swing and stance

max_dt = 0.035
min_dt = 0.01

# min_jump_height = 0.5
# #decide time step lengths for swing and stance

dt = prog.NewContinuousVariables(1,"dt")

#decide state positions 
q = prog.NewContinuousVariables(number_of_collocation_points, 4, 'q')
#decide state velocities
qd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qd')
#decide state accelerations
qdd = prog.NewContinuousVariables(number_of_collocation_points, 4, 'qdd')
#decide leg input torques
u = prog.NewContinuousVariables(number_of_collocation_points, 1, 'u') 

prog.AddLinearConstraint( q[0,0] == q_initial_state[0]).evaluator().set_description('x_init') #x position
prog.AddLinearConstraint( q[0,1] <= q_initial_state[1]).evaluator().set_description('z_init') #z position
prog.AddLinearConstraint( q[0,2] == q_initial_state[2]).evaluator().set_description('l_init') #l length
prog.AddLinearConstraint( q[0,3] <= q_initial_state[3]).evaluator().set_description('theta_init') #theta angle

#prog.AddLinearConstraint( q[-2,0] >= q_final_state[0]).evaluator().set_description('x_final') #x position
#prog.AddLinearConstraint( q[-2,1] >= q_final_state[1]).evaluator().set_description('z_final') #z position
#prog.AddLinearConstraint( q[-2,2] >= q_final_state[2]).evaluator().set_description('l_final') #l length
#prog.AddLinearConstraint( q[-1,3] == q_final_state[3]).evaluator().set_description('theta_final') # theta length

#velocity

prog.AddLinearConstraint( qd[0,0] == q_dot_initial_state[0] ).evaluator().set_description('xd_init')  #x vel
prog.AddLinearConstraint( qd[0,1] <= q_dot_initial_state[1] ).evaluator().set_description('zd_init')  #z vel
prog.AddLinearConstraint( qd[0,2] <= q_dot_initial_state[2] ).evaluator().set_description('ld_init')  #l vel
#prog.AddLinearConstraint( qd[0,3] <= q_dot_initial_state[3] ).evaluator().set_description('thetad_init')  #theta vel

#prog.AddLinearConstraint( qd[-2,0] >= q_dot_final_state[0] ).evaluator().set_description('xd_end')  #x vel
#prog.AddLinearConstraint( qd[-2,1] >= q_dot_final_state[1] ).evaluator().set_description('zd_end')  #z vel
# prog.AddLinearConstraint( qd[-1,2] == q_dot_final_state[2] ).evaluator().set_description('thetad_end')  #l vel

#Actuator spring constant constraint
prog.AddBoundingBoxConstraint([slip_model.min_spring_const]*number_of_collocation_points, 
                              [slip_model.max_spring_const]*number_of_collocation_points,
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
    prog.AddConstraint(slip_model.get_foot_z_position,lb=[0], ub=[0],vars=var).evaluator().set_description("grounded foot")
    prog.AddLinearConstraint(q[i,1]>=0.2).evaluator().set_description("min stance height")
    prog.AddConstraint(slip_model.stance_feasibility, lb=[0]*2,ub=[0]*2,vars=var).evaluator().set_description("stance eqn")



    if i < number_of_collocation_points - 1:
        prog.AddConstraint(eq(q[i+1,:], q[i,:] + dt * qd[i,:])).evaluator().set_description('stance full state euler integration pos')
        prog.AddConstraint(eq(qd[i+1,:], qd[i,:] + dt * qdd[i,:])).evaluator().set_description('stance full state euler integration vel')

    else:
        prog.AddConstraint(qd[i,0]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[0])
        prog.AddConstraint(qd[i,1]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[1])
        prog.AddConstraint(qdd[i,0]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[2])
        prog.AddConstraint(qdd[i,1]==slip_model.relate_pelvis_accel_to_leg_accel(q[i,:],qd[i,:],qdd[i,:])[3])
        prog.AddLinearConstraint(q[i,2] == slip_model.l_rest)


   
    #prog.AddCost(dt[0]*u[i].dot(u[i]))

solver = SnoptSolver()
result = solver.Solve(prog)

pos = result.GetSolution(q)
vel = result.GetSolution(qd)
accel = result.GetSolution(qdd)
inputs = result.GetSolution(u)


#plt.plot(pos[:,0] - pos[:,2] * np.sin(pos[:,3]), label='foot x pos')
#plt.plot(pos[:,0], label='hip x pos')
#plt.legend()
#plt.show()

plt.plot(pos[:-1,0],label = "x distance")
plt.plot(pos[:-1,1],label = "z height")
plt.plot(pos[:-1,2],label = "leg length")
plt.plot(pos[:-1,3],label = "leg angle")
plt.plot(vel[:,0][:-1],label = "x vel")
plt.plot(vel[:,1][:-1],label = "z vel")
plt.legend()
plt.show()


if not result.is_success():
    print(result.GetInfeasibleConstraintNames(prog))

    raise RuntimeError("Could not generate trajectory")

slip_model.animate(pos)

