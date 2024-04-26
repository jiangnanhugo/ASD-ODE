def sympy_plus_scipy():
    from sympy import symbols, lambdify
    import numpy as np
    import scipy.integrate

    # Create symbols y0, y1, and y2
    y = symbols('y:3')

    rf = y[0] ** 2 * y[1]
    rb = y[2] ** 2
    # Derivative of the function y(t); values for the three chemical species
    # for input values y, kf, and kb
    ydot = [2 * (rb - rf), rb - rf, 2 * (rf - rb)]
    print(ydot)
    t = symbols('t')  # not used in this case
    # Convert the SymPy symbolic expression for ydot into a form that
    # SciPy can evaluate numerically, f
    f = lambdify((t, y), ydot)
    k_vals = np.array([0.42, 0.17])  # arbitrary in this case
    y0 = [1, 0, 1]  # initial condition (initial values)
    y0 = np.asarray(y0)
    y0 = y0.T
    print(y0.shape)
    t_eval = np.linspace(0, 10, 50)  # evaluate integral from t = 0-10 for 50 points
    # Call SciPy's ODE initial value problem solver solve_ivp by passing it
    #   the function f,
    #   the interval of integration,
    #   the initial state, and
    #   the arguments to pass to the function f
    solution = scipy.integrate.solve_ivp(f, (0, 10), y0, t_eval=t_eval, vectorized=True)
    # Extract the y (concentration) values from SciPy solution result
    y = solution.y
    print(y.shape)
    # Plot the result graphically using matplotlib
