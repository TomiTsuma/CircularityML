import sys

size = "Not Set"
points = 10

def reward(width, length , mass):
    area = width * length *100
    size = "Not Set"
    points = 0.29


    ##################Change the first condition to check if area==0 in which case points = 0
    if area==0:
        points = 0
    elif area <= 30:
        size = "350ml"
        if mass > 20:
            sys.stdout.write("mass = "+str(mass))
            points = 0
    elif area <= 45:
        size = "500ml"
        if mass > 20:
            sys.stdout.write("mass = "+str(mass))
            points = points=0
    else:
        size = "1 Litre"
        if mass > 20:
            sys.stdout.write("mass = "+str(mass))
            points = points =0
    sys.stdout.write(size)
    sys.stdout.write("mass = "+str(mass))
    sys.stdout.write(("points = "+str(points)+"\n"))
    sys.stdout.write(str(area))

    return points
