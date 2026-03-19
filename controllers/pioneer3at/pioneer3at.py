from controller import Robot, Supervisor, DistanceSensor
import grid
import estimator

TIME_STEP = 32
DIST_THRESHOLD = 800.0;
MAX_SPEED = 6.0;
TURN_SPEED = 3.0;

robot = Supervisor()
robotNode = robot.getSelf()
timeStep = int(robot.getBasicTimeStep())

## Robot initialization 
blwheel = robot.getDevice("back left wheel")
brwheel = robot.getDevice("back right wheel")
flwheel = robot.getDevice("front left wheel")
frwheel = robot.getDevice("front right wheel")

s0 = robot.getDevice("so0") #Side Left
s1 = robot.getDevice("so1") #Left
s2 = robot.getDevice("so2") #Midde Left
s3 = robot.getDevice("so3") #Front Left
s4 = robot.getDevice("so4") #Front Right
s5 = robot.getDevice("so5") #Middle Right
s6 = robot.getDevice("so6") #Right
s7 = robot.getDevice("so7") #Side Right
s0.enable(timeStep)
s1.enable(timeStep)
s2.enable(timeStep)
s3.enable(timeStep)
s4.enable(timeStep)
s5.enable(timeStep)
s6.enable(timeStep)
s7.enable(timeStep)

blwheel.setPosition(float('inf'))
brwheel.setPosition(float('inf'))
flwheel.setPosition(float('inf'))
frwheel.setPosition(float('inf'))

blwheel.setVelocity(0)
brwheel.setVelocity(0)
flwheel.setVelocity(0)
frwheel.setVelocity(0)


## Variables
prevPosition = [0,0]
grid = grid.Grid()
estimator = estimator.Estimator(grid.grid_gain, prevPosition, grid.nn, grid.mm)

## Let the grid cells network stabilize
for i in range(100):
    if i < 25:
        speed = 1 + 0j
    elif i < 50:
        speed = 0 + 1j
    elif i < 75:
        speed = -1 + 0j
    else:
        speed = 0 - 1j
    grid.update(speed)


while robot.step(TIME_STEP) != -1:
    
    """ BRAITENBERG -----------------"""
    # Get values from distance sensors
    s0_val = s0.getValue()
    s1_val = s1.getValue()
    s2_val = s2.getValue()
    s3_val = s3.getValue()
    s4_val = s4.getValue()
    s5_val = s5.getValue()
    s6_val = s6.getValue()
    s7_val = s7.getValue()
    
    # Change wheels velocity
    if(s3_val >= DIST_THRESHOLD) & (s4_val >= DIST_THRESHOLD):
        blwheel.setVelocity(-TURN_SPEED)
        brwheel.setVelocity(-TURN_SPEED)
        flwheel.setVelocity(-TURN_SPEED)
        frwheel.setVelocity(-TURN_SPEED)
    elif(s0_val >= DIST_THRESHOLD) | (s1_val >= DIST_THRESHOLD) | (s2_val >= DIST_THRESHOLD) | (s3_val >= DIST_THRESHOLD):
        blwheel.setVelocity(TURN_SPEED)
        brwheel.setVelocity(-TURN_SPEED)
        flwheel.setVelocity(TURN_SPEED)
        frwheel.setVelocity(-TURN_SPEED)
    elif(s4_val >= DIST_THRESHOLD) | (s5_val >= DIST_THRESHOLD) | (s6_val >= DIST_THRESHOLD) | (s7_val >= DIST_THRESHOLD):
        blwheel.setVelocity(-TURN_SPEED)
        brwheel.setVelocity(TURN_SPEED)
        flwheel.setVelocity(-TURN_SPEED)
        frwheel.setVelocity(TURN_SPEED)
    else:
        blwheel.setVelocity(MAX_SPEED)
        brwheel.setVelocity(MAX_SPEED)
        flwheel.setVelocity(MAX_SPEED)
        frwheel.setVelocity(MAX_SPEED)
    
    """ END OF BRAITENBERG -------------"""
    
    # Get the current grid state
    start_conf = grid.grid_activity.copy()
    
    ## Get velocity -----------------     
    pos = robotNode.getPosition()
    velx = round(pos[0], 3) - prevPosition[0]
    vely = round(pos[1], 3) - prevPosition[1]
    speed = velx + 1j * vely
    prevPosition[0] = round(pos[0], 3)
    prevPosition[1] = round(pos[1], 3)
    ## ------------------------------   
    
    # Update grid network based on new speed
    grid.update(speed)
    
    # Get the new grid state
    end_conf = grid.grid_activity.copy()
    
    # Update estimation
    # Output : Estimated X, Estimated Y, Network information (ignored)
    new_x, new_y, info = estimator.new_estimation(start_conf, end_conf)
    
    print("Current position - X:{:.2f} , Y:{:.2f}".format(pos[0], pos[1]))
    print("Estimated position - X:{:.2f} , Y:{:.2f}".format(new_x, new_y))
        