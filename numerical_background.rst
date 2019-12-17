The hillslope diffustion equation is:

.. math:: \frac{\partial z}{\partial t} = D \nabla^2 z = D \bigg( \frac{\partial^2 z}{\partial x^2} + \frac{\partial^2 z}{\partial y^2} \bigg)

 By taking the Fourier transform of equation (1) in both x and y
directions, we obtain the spectral form of this problem for wavenumbers
:math:`k_x` and :math:`k_y`:

.. math:: \frac{\partial \hat{\eta}_{k_x,k_y}}{\partial t} = D \big(-k_x^2 \hat{\eta}_{k_x,k_y}  -k_y^2 \hat{\eta}_{k_x,k_y} \big)

 where the wavenumber :math:`k_x` is:

.. math:: k_x = \frac{2\pi n_x}{N_x \Delta x}, \hspace{1cm} n_x = [-N_x/2,N_x/2]

| where :math:`N_x` is the dimension of the grid in the x direction.
| The **explicit form** of this equation considers the right hand side
  of the equation only at time step n:

  .. math:: \frac{\hat{\eta}^{n+1}_{k_x,k_y} - \hat{\eta}^{n}_{k_x,k_y}} {\Delta t} = D \big(-k_x^2 \hat{\eta}^{n}_{k_x,k_y}  -k_y^2 \hat{\eta}^{n}_{k_x,k_y} \big)

   The **implicit (Crank-Nicholson) form** of this equation uses both
  time :math:`n` and time :math:`n+1` on the right side, and does not
  have the stability constraint of the explicit form:

  .. math:: \frac{\hat{\eta}^{n+1}_{k_x,k_y} - \hat{\eta}^{n}_{k_x,k_y}} {\Delta t} = \frac{D}{2} \big(-k_x^2 \hat{\eta}^{n}_{k_x,k_y}  -k_y^2 \hat{\eta}^{n}_{k_x,k_y} -k_x^2 \hat{\eta}^{n+1}_{k_x,k_y}  -k_y^2 \hat{\eta}^{n+1}_{k_x,k_y}\big)

For comparison, a simple explicit method often used to solve the
diffusion equation is:

.. math:: \frac{\eta_{i,j}^{n+1} - \eta_{i,j}^{n}}{\Delta t} = D\bigg( \frac{\eta_{i-1,j}^n - 2 \eta_{i,j}^n + \eta_{i+1,j}^n}{\Delta x^2} + \frac{\eta_{i,j-1}^n - 2 \eta_{i,j}^n + \eta_{i,j+1}^n}{\Delta y^2} \bigg)
