import math
from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg as linalg
from scipy.stats import norm

def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Price a European option using the Black-Scholes model.

    Args:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to maturity (in years)
        r (float): Risk-free interest rate
        sigma (float): Volatility of the stock
        option_type (str, optional): 'call' or 'put'. Defaults to 'call'.

    Returns:
        float: Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    elif option_type.lower() == 'put':
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    return price


class StockOption(object):
    """ 
    Stores common attributes of a stock option 
    """
    def __init__(
        self, S0, K, r=0.05, T=1, N=2, pu=0, pd=0, 
        div=0, sigma=0, is_put=False, is_am=False):
        """
        Initialize the stock option base class.
        Defaults to European call unless specified.

        :param S0: initial stock price
        :param K: strike price
        :param r: risk-free interest rate
        :param T: time to maturity
        :param N: number of time steps
        :param pu: probability at up state
        :param pd: probability at down state
        :param div: Dividend yield
        :param is_put: True for a put option,
                False for a call option
        :param is_am: True for an American option,
                False for a European option
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.N = max(1, N)
        self.STs = [] # Declare the stock prices tree

        """ Optional parameters used by derived classes """
        self.pu, self.pd = pu, pd
        self.div = div
        self.sigma = sigma
        self.is_call = not is_put
        self.is_european = not is_am

    @property
    def dt(self):
        """ Single time step, in years """
        return self.T/float(self.N)

    @property
    def df(self):
        """ The discount factor """
        return math.exp(-(self.r-self.div)*self.dt)  


class BinomialEuropeanOption(StockOption):
    """ 
    Price a European option by the binomial tree model 
    """
    def setup_parameters(self):
        # Required calculations for the model
        self.M = self.N+1  # Number of terminal nodes of tree
        self.u = 1+self.pu  # Expected value in the up state
        self.d = 1-self.pd  # Expected value in the down state
        self.qu = (math.exp(
            (self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
        self.qd = 1-self.qu

    def init_stock_price_tree(self):
        # Initialize terminal price nodes to zeros
        self.STs = np.zeros(self.M)

        # Calculate expected stock prices for each node
        for i in range(self.M):
            self.STs[i] = self.S0 * \
                (self.u**(self.N-i)) * (self.d**i)

    def init_payoffs_tree(self):
        """
        Returns the payoffs when the option 
        expires at terminal nodes
        """ 
        if self.is_call:
            return np.maximum(0, self.STs-self.K)
        else:
            return np.maximum(0, self.K-self.STs)

    def traverse_tree(self, payoffs):
        """
        Starting from the time the option expires, traverse
        backwards and calculate discounted payoffs at each node
        """
        for i in range(self.N):
            payoffs = (payoffs[:-1]*self.qu + 
                       payoffs[1:]*self.qd)*self.df

        return payoffs

    def begin_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self):
        """ Entry point of the pricing implementation """
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()

        # Option value converges to first node
        return payoffs[0]


class BinomialTreeOption(StockOption):
    """ 
    Price a European or American option by the binomial tree 
    """
    def setup_parameters(self):
        self.u = 1+self.pu  # Expected value in the up state
        self.d = 1-self.pd  # Expected value in the down state
        self.qu = (math.exp(
            (self.r-self.div)*self.dt)-self.d)/(self.u-self.d)
        self.qd = 1-self.qu

    def init_stock_price_tree(self):
        # Initialize a 2D tree at T=0
        self.STs = [np.array([self.S0])]

        # Simulate the possible stock prices path
        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate(
                (prev_branches*self.u, 
                 [prev_branches[-1]*self.d]))
            self.STs.append(st) # Add nodes at each time step

    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, self.STs[self.N]-self.K)
        else:
            return np.maximum(0, self.K-self.STs[self.N])

    def check_early_exercise(self, payoffs, node):
        if self.is_call:
            return np.maximum(payoffs, self.STs[node] - self.K)
        else:
            return np.maximum(payoffs, self.K - self.STs[node])

    def traverse_tree(self, payoffs):
        for i in reversed(range(self.N)):
            # The payoffs from NOT exercising the option
            payoffs = (payoffs[:-1]*self.qu + 
                       payoffs[1:]*self.qd)*self.df

            # Payoffs from exercising, for American options
            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs,i)

        return payoffs

    def begin_tree_traversal(self):
        payoffs = self.init_payoffs_tree()
        return self.traverse_tree(payoffs)

    def price(self):
        """  The pricing implementation """
        self.setup_parameters()
        self.init_stock_price_tree()
        payoffs = self.begin_tree_traversal()
        return payoffs[0]


class BinomialCRROption(BinomialTreeOption):
    """ 
    Price an option by the binomial CRR model 
    """
    def setup_parameters(self):
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1./self.u
        self.qu = (math.exp((self.r-self.div)*self.dt) - 
                   self.d)/(self.u-self.d)
        self.qd = 1-self.qu


class BinomialLROption(BinomialTreeOption):
    """ 
    Price an option by the Leisen-Reimer tree
    """
    def setup_parameters(self):
        odd_N = self.N if (self.N%2 == 0) else (self.N+1)
        d1 = (math.log(self.S0/self.K) +
              ((self.r-self.div) +
               (self.sigma**2)/2.)*self.T)/\
            (self.sigma*math.sqrt(self.T))
        d2 = (math.log(self.S0/self.K) +
              ((self.r-self.div) -
               (self.sigma**2)/2.)*self.T)/\
            (self.sigma * math.sqrt(self.T))

        pbar = self.pp_2_inversion(d1, odd_N)
        self.p = self.pp_2_inversion(d2, odd_N)
        self.u = 1/self.df * pbar/self.p
        self.d = (1/self.df-self.p*self.u)/(1-self.p)
        self.qu = self.p
        self.qd = 1-self.p

    def pp_2_inversion(self, z, n):
        return .5 + math.copysign(1, z)*\
            math.sqrt(.25 - .25*
                math.exp(
                    -((z/(n+1./3.+.1/(n+1)))**2.)*(n+1./6.)
                )
            )


class BinomialLRWithGreeks(BinomialLROption):
    """ 
    Compute option price, delta and gamma by the LR tree 
    """
    def new_stock_price_tree(self):
        """
        Creates an additional layer of nodes to our
        original stock price tree
        """
        self.STs = [np.array([self.S0*self.u/self.d,
                              self.S0,
                              self.S0*self.d/self.u])]

        for i in range(self.N):
            prev_branches = self.STs[-1]
            st = np.concatenate((prev_branches*self.u,
                                 [prev_branches[-1]*self.d]))
            self.STs.append(st)

    def price(self):
        self.setup_parameters()
        self.new_stock_price_tree()
        payoffs = self.begin_tree_traversal()

        # Option value is now in the middle node at t=0
        option_value = payoffs[len(payoffs)//2]

        payoff_up = payoffs[0]
        payoff_down = payoffs[-1]
        S_up = self.STs[0][0]
        S_down = self.STs[0][-1]
        dS_up = S_up - self.S0
        dS_down = self.S0 - S_down

        # Calculate delta value
        dS = S_up - S_down
        dV = payoff_up - payoff_down
        delta = dV/dS

        # calculate gamma value
        gamma = ((payoff_up-option_value)/dS_up - 
                 (option_value-payoff_down)/dS_down) / \
            ((self.S0+S_up)/2. - (self.S0+S_down)/2.)

        return option_value, delta, gamma


class TrinomialTreeOption(BinomialTreeOption):

    def setup_parameters(self):
        """ Required calculations for the model """
        self.u = math.exp(self.sigma*math.sqrt(2.*self.dt))
        self.d = 1/self.u
        self.m = 1
        self.qu = ((math.exp((self.r-self.div) *
                             self.dt/2.) -
                    math.exp(-self.sigma *
                             math.sqrt(self.dt/2.))) /
                   (math.exp(self.sigma *
                             math.sqrt(self.dt/2.)) -
                    math.exp(-self.sigma *
                             math.sqrt(self.dt/2.))))**2
        self.qd = ((math.exp(self.sigma *
                             math.sqrt(self.dt/2.)) -
                    math.exp((self.r-self.div) *
                             self.dt/2.)) /
                   (math.exp(self.sigma *
                             math.sqrt(self.dt/2.)) -
                    math.exp(-self.sigma *
                             math.sqrt(self.dt/2.))))**2.

        self.qm = 1 - self.qu - self.qd

    def init_stock_price_tree(self):
        # Initialize a 2D tree at t=0
        self.STs = [np.array([self.S0])]

        for i in range(self.N):
            prev_nodes = self.STs[-1]
            self.ST = np.concatenate(
                (prev_nodes*self.u, [prev_nodes[-1]*self.m,
                                     prev_nodes[-1]*self.d]))
            self.STs.append(self.ST)

    def traverse_tree(self, payoffs):
        # Traverse the tree backwards 
        for i in reversed(range(self.N)):
            payoffs = (payoffs[:-2] * self.qu +
                       payoffs[1:-1] * self.qm +
                       payoffs[2:] * self.qd) * self.df

            if not self.is_european:
                payoffs = self.check_early_exercise(payoffs,i)

        return payoffs


class BinomialCRRLattice(BinomialCRROption):

    def setup_parameters(self):
        super(BinomialCRRLattice, self).setup_parameters()
        self.M = 2*self.N + 1

    def init_stock_price_tree(self):
        self.STs = np.zeros(self.M)
        self.STs[0] = self.S0 * self.u**self.N

        for i in range(self.M)[1:]:
            self.STs[i] = self.STs[i-1]*self.d

    def init_payoffs_tree(self):
        odd_nodes = self.STs[::2]  # Take odd nodes only
        if self.is_call:
            return np.maximum(0, odd_nodes-self.K)
        else:
            return np.maximum(0, self.K-odd_nodes)

    def check_early_exercise(self, payoffs, node):
        self.STs = self.STs[1:-1]  # Shorten ends of the list
        odd_STs = self.STs[::2]  # Take odd nodes only
        if self.is_call:
            return np.maximum(payoffs, odd_STs-self.K)
        else:
            return np.maximum(payoffs, self.K-odd_STs)


class TrinomialLattice(TrinomialTreeOption):
    """ 
    Price an option by the trinomial lattice 
    """
    def setup_parameters(self):
        super(TrinomialLattice, self).setup_parameters()
        self.M = 2*self.N + 1

    def init_stock_price_tree(self):
        self.STs = np.zeros(self.M)
        self.STs[0] = self.S0 * self.u**self.N

        for i in range(self.M)[1:]:
            self.STs[i] = self.STs[i-1]*self.d

    def init_payoffs_tree(self):
        if self.is_call:
            return np.maximum(0, self.STs-self.K)
        else:
            return np.maximum(0, self.K-self.STs)

    def check_early_exercise(self, payoffs, node):
        self.STs = self.STs[1:-1]  # Shorten ends of the list
        if self.is_call:
            return np.maximum(payoffs, self.STs-self.K)
        else:
            return np.maximum(payoffs, self.K-self.STs)


class FiniteDifferences(object):
    """ 
    Base class for sharing 
    attributes and functions of FD 
    """
    def __init__(
        self, S0, K, r=0.05, T=1, 
        sigma=0, Smax=1, M=1, N=1, is_put=False
    ):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.Smax = Smax
        self.M, self.N = M, N
        self.is_call = not is_put

        self.i_values = np.arange(self.M)
        self.j_values = np.arange(self.N)
        self.grid = np.zeros(shape=(self.M+1, self.N+1))
        self.boundary_conds = np.linspace(0, Smax, self.M+1)

    @property
    def dS(self):
        return self.Smax/float(self.M)

    @property
    def dt(self):
        return self.T/float(self.N)

    @abstractmethod
    def setup_boundary_conditions(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def setup_coefficients(self):
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def traverse_grid(self):
        """  
        Iterate the grid backwards in time 
        """
        raise NotImplementedError('Implementation required!')

    @abstractmethod
    def interpolate(self):
        """
        Use piecewise linear interpolation on the initial
        grid column to get the closest price at S0.
        """
        return np.interp(
            self.S0, self.boundary_conds, self.grid[:,0])

    def price(self):
        self.setup_boundary_conditions()
        self.setup_coefficients()
        self.traverse_grid()
        return self.interpolate()


class FDExplicitEu(FiniteDifferences):
    """ 
    Explicit method of Finite Differences 
    """
    def setup_boundary_conditions(self):
        if self.is_call:
            self.grid[:,-1] = np.maximum(
                0, self.boundary_conds - self.K)
            self.grid[-1,:-1] = (self.Smax-self.K) * \
                np.exp(-self.r*self.dt*(self.N-self.j_values))
        else:
            self.grid[:,-1] = np.maximum(
                0, self.K-self.boundary_conds)
            self.grid[0,:-1] = (self.K-self.Smax) * \
                np.exp(-self.r*self.dt*(self.N-self.j_values))

    def setup_coefficients(self):
        self.a = 0.5*self.dt*((self.sigma**2) *
                              (self.i_values**2) -
                              self.r*self.i_values)
        self.b = 1 - self.dt*((self.sigma**2) *
                              (self.i_values**2) +
                              self.r)
        self.c = 0.5*self.dt*((self.sigma**2) *
                              (self.i_values**2) +
                              self.r*self.i_values)

    def traverse_grid(self):
        for j in reversed(self.j_values):
            for i in range(self.M)[2:]:
                self.grid[i,j] = \
                    self.a[i]*self.grid[i-1,j+1] +\
                    self.b[i]*self.grid[i,j+1] + \
                    self.c[i]*self.grid[i+1,j+1]


class FDImplicitEu(FDExplicitEu):
    """ 
    Explicit method of Finite Differences 
    """
    def setup_coefficients(self):
        self.a = 0.5*(self.r*self.dt*self.i_values -
                      (self.sigma**2)*self.dt*\
                          (self.i_values**2))
        self.b = 1 + \
                 (self.sigma**2)*self.dt*\
                    (self.i_values**2) + self.r*self.dt
        self.c = -0.5*(self.r*self.dt*self.i_values +
                       (self.sigma**2)*self.dt*\
                           (self.i_values**2))
        self.coeffs = np.diag(self.a[2:self.M],-1) + \
                      np.diag(self.b[1:self.M]) + \
                      np.diag(self.c[1:self.M-1],1)

    def traverse_grid(self):
        """ Solve using linear systems of equations """
        P, L, U = linalg.lu(self.coeffs)
        aux = np.zeros(self.M-1)

        for j in reversed(range(self.N)):
            aux[0] = np.dot(-self.a[1], self.grid[0, j])
            x1 = linalg.solve(L, self.grid[1:self.M, j+1]+aux)
            x2 = linalg.solve(U, x1)
            self.grid[1:self.M, j] = x2


class FDCnEu(FDExplicitEu):
    """ 
    Crank-Nicolson method of Finite Differences 
    """
    def setup_coefficients(self):
        self.alpha = 0.25*self.dt*(
            (self.sigma**2)*(self.i_values**2) - \
            self.r*self.i_values)
        self.beta = -self.dt*0.5*(
            (self.sigma**2)*(self.i_values**2) + self.r)
        self.gamma = 0.25*self.dt*(
            (self.sigma**2)*(self.i_values**2) +
            self.r*self.i_values)
        self.M1 = -np.diag(self.alpha[2:self.M], -1) + \
                  np.diag(1-self.beta[1:self.M]) - \
                  np.diag(self.gamma[1:self.M-1], 1)
        self.M2 = np.diag(self.alpha[2:self.M], -1) + \
                  np.diag(1+self.beta[1:self.M]) + \
                  np.diag(self.gamma[1:self.M-1], 1)

    def traverse_grid(self):
        """ Solve using linear systems of equations """
        P, L, U = linalg.lu(self.M1)

        for j in reversed(range(self.N)):
            x1 = linalg.solve(
                L, np.dot(self.M2, self.grid[1:self.M, j+1]))
            x2 = linalg.solve(U, x1)
            self.grid[1:self.M, j] = x2

