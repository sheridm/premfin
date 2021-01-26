from pandas import read_excel, Series
from numpy import cumprod
from pickle import load
from CONST import *

vbt = load(open(DATA_FOLDER + 'vbt', 'rb'))

# need to `pip install openpyxl`
malevbt = read_excel(
    vbt, sheet_name='2015 Male Unismoke ANB', header=2, index_col=0
)
femavbt = read_excel(
    r.content, sheet_name='2015 Female Unismoke ANB', header=2, index_col=0
)


# survival curve
def surcurv(isMale: bool, age: int):
    tbl = malevbt if isMale else femavbt
    maxage = max(tbl.index)
    if age <= maxage:
        curv = tbl.loc[age][:25].append(tbl['Ult.'][age:])
    else:
        curv = tbl.loc[maxage][(age-maxage):26]
    mort = Series([0]).append(curv, ignore_index=True)
    surv = cumprod(1-mort/1000)
    return surv


def premium_cf(surv, pr, r_free: float = 0.1) -> float:
    cf = 0

    if isinstance(pr, (int, float)):
        for i, s in enumerate(surv):
            cf += surv[i]/(1+r_free)**i
        cf *= pr

    else:
        assert len(surv) == len(
            pr), 'survial curve and premium curve must have the same length'
        for i, s in enumerate(surv):
            cf += (surv[i] * pr[i])/(1+r_free)**i

    return cf


def pv_premium(pr, r_free: float = 0.1) -> float:
    # entry = nquad(entryint, [
    #     lambda phi, tee: [lstar(tee, phi), np.inf],
    #     [0, upperbound],
    #     [0, np.inf]
    # ], opts=[quadoptions, quadoptions, quadoptions])[0]

    # return dict(plotdata=df, entrydata=entry)


if __name__ == "__main__":
