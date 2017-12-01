# use simpson's rule to integrate x^4 - 2x + 1 from 0 to 2

def integrateSimpson(func, a, b, slices):
    step = (b - a) / slices
    area = 0
    for i in range(slices):
        x1 = a + i*step
        x3 = a + (i+1)*step
        x2 = (x1 + x3) / 2.0
        # simpson's rule approximates a quadratic through 3 points (x1, x2 and x3) and computes the volume under
        # the quadratic. It turns out* that int_x1^x3 ~ (x3-x1)/6 * (f(x1) + 4f(x2) + f(x3))
        area += (x3 - x1)*(func(x1) + func(x3) + 4*func(x2)) / 6.0
    return area

func = lambda x: x**4 - 2*x + 1
for j in range(1,5):
    slices = 10**j
    res = integrateSimpson(func, 0, 2, slices)
    print('slices: ',slices)
    print(res)
    print('fractional error: ', str((res-4.4)/4.4))
    print('\n')

# *"turns out" is a bit misleading. Any (definite) integral is a linear transformation and thus has a matrix. For any
# function, there exists a lagrange interpolation polynomial with n points, and it follows that any definite integral is a linear
# combination of the polynomial evaluated at those n points
