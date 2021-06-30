from dataclasses import dataclass
from re import X
import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
from typing import Optional

from scipy import optimize

from premiumFinance.constants import DATA_FOLDER
from premiumFinance.settings import PROJECT_ROOT
from premiumFinance.fetchdata import getVBTdata

# from sympy.solvers import solve
# from sympy import Symbol

EPSILON = 1e-10


@dataclass
class Mortality:
    issueage: int
    currentage: Optional[int]
    isMale: bool
    isSmoker: Optional[bool]
    mortrate: float = 1
    whichVBT: str = "VBT01"

    def __post_init__(self):
        if self.currentage is None:
            self.currentage = self.issueage
        assert self.issueage <= self.currentage, "Issue age must not exceed current age"

    @property
    def gender(self):
        mf = "Male" if self.isMale else "Female"
        return mf

    @property
    def basemortCurv(self):
        mort = getVBTdata(
            vbt=self.whichVBT,
            isMale=self.isMale,
            isSmoker=self.isSmoker,
            issueage=self.issueage,
            currentage=self.currentage,
        )
        return mort

    @property
    def conditional_mortality_curve(self):
        mort = self.basemortCurv
        # adjust mortality rate with multiplier
        condMort = pd.Series(min(1 - EPSILON, self.mortrate * q) for q in mort)
        return condMort

    @property
    def conditional_survival_curve(self):
        condMort = self.conditional_mortality_curve
        condSurv = 1 - condMort
        return condSurv

    @property
    def survCurv(self):
        condSurv = self.conditional_survival_curve
        surv = np.cumprod(condSurv)
        return surv

    @property
    def lifeExpectancy(self):
        surv = self.survCurv
        le = np.sum(surv) - 0.5
        return le

    def plotSurvCurv(self):
        surv = self.survCurv
        le = self.lifeExpectancy
        leage = le + self.currentage
        plt.plot(surv.index + self.currentage, surv, label="Survival rate")
        plt.xlabel("Age")
        plt.ylabel("Cumulative probability")
        plt.axvline(leage, color="red", label=f"LE: {round(leage,1)}")
        plt.title(
            f"issue age: {self.issueage}, gender: {self.gender}, smoker: {self.isSmoker}"
        )
        plt.axvline(self.currentage, ls="--", lw=0.5, color="gray")
        plt.axhline(0, ls="--", lw=0.5, color="gray")
        plt.axhline(1, ls="--", lw=0.5, color="gray")
        plt.legend()


def calc_le(age: int, is_male: bool, is_smoker: bool, mortrate: float) -> float:
    mortality = Mortality(
        issueage=age,
        currentage=age,
        isMale=is_male,
        isSmoker=is_smoker,
        mortrate=mortrate,
        whichVBT="VBT15",
    )
    return mortality.lifeExpectancy


def implied_mortality(
    age: int, is_male: bool, is_smoker: bool, target_le: float
) -> float:
    x = optimize.root_scalar(
        lambda x: calc_le(
            age=age,
            is_male=is_male,
            is_smoker=is_smoker,
            mortrate=x,
        )
        - target_le,
        x0=1,
        bracket=[0.001, 3],
        method="brentq",
    )
    return x.root


#
