import requests
import pickle
import constants
import csv
import pandas as pd
from io import TextIOWrapper, BytesIO
from zipfile import ZipFile
from pprint import pprint
import xml.etree.ElementTree as ET
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# retrieve VBT data
def getVBTdata(url: str = constants.VBT_URL):
    r_vbt = requests.get(url)
    pickle.dump(r_vbt.content, open(constants.DATA_FOLDER + 'vbt', 'wb'))


def getYieldData(rooturl: str = constants.YIELD_URL, entryindex: int = 7782, month: int = 2, year: int = 2021):
    if entryindex is None:
        url = f'{rooturl}?$filter=month(NEW_DATE) eq {month} and year(NEW_DATE) eq {year}'
    url = f'{rooturl}({entryindex})'
    r_yield = requests.get(url)
    content = r_yield.content.decode("utf-8")
    root = ET.fromstring(content)
    yieldTable = [{'duration': 0, 'rate': 0}]

    yieldTable.extend({
        'duration': constants.YIELD_DURATION[w.tag[58:]],
        'rate': float(w.text)/100
    }for w in root[6][0][2:-1]
    )

    return pd.DataFrame(yieldTable)


# kind has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
# or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth, first, second or third order;
# ‘previous’ and ‘next’ simply return the previous or next value of the point;
# ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down.
def getAnnualYield(yieldTable=None, durange=range(150), intertype: str = 'linear'):
    if yieldTable is None:
        yieldTable = getYieldData()
    f = interp1d(yieldTable['duration'], yieldTable['rate'],
                 kind=intertype,
                 fill_value=tuple(yieldTable.iloc[[0, -1]]['rate']),
                 bounds_error=False
                 )
    return f(durange)

# retrieve the huge mortality data set from the SOA


def getMortData(url: str = constants.MORT_URL):
    r_mort = requests.get(url)

    zip_ref = ZipFile(BytesIO(r_mort.content))

    for i, name in enumerate(zip_ref.namelist()):
        # to make sure there is only one file in the zip
        print(str(i)+name)
        with zip_ref.open(name) as file_contents:
            reader = csv.DictReader(TextIOWrapper(
                file_contents), delimiter='\t')
            for j, item in enumerate(reader):
                # try a few rows
                if j > 1:
                    break
                print(str(j) + '=========')
                pprint(item)
                # {'Age Basis': '0',
                #  'Amount Exposed': '2742585.841000',
                #  'Attained Age': '52',
                #  'Common Company Indicator 57': '1',
                #  'Death Claim Amount': '.000000',
                #  'Duration': '9',
                #  'Expected Death QX2001VBT by Amount': '5978.8371333800014',
                #  'Expected Death QX2001VBT by Policy': '4.4306527100000007E-2',
                #  'Expected Death QX2008VBT by Amount': '3675.0650269400003',
                #  'Expected Death QX2008VBT by Policy': '2.7234287300000003E-2',
                #  'Expected Death QX2008VBTLU by Amount': '6582.2060183999984',
                #  'Expected Death QX2008VBTLU by Policy': '4.8777828000000002E-2',
                #  'Expected Death QX2015VBT by Amount': '2989.4185666900007',
                #  'Expected Death QX2015VBT by Policy': '2.2153263550000003E-2',
                #  'Expected Death QX7580E by Amount': '8803.700549610001',
                #  'Expected Death QX7580E by Policy': '6.5240344949999973E-2',
                #  'Face Amount Band': '  100000-249999',
                #  'Gender': 'Female',
                #  'Insurance Plan': ' Term',
                #  'Issue Age': '44',
                #  'Issue Year': '2000',
                #  'Number of Deaths': '0',
                #  'Number of Preferred Classes': '2',
                #  'Observation Year': '2009',
                #  'Policies Exposed': '20.324095',
                #  'Preferred Class': '2',
                #  'Preferred Indicator': '1',
                #  'SOA Anticipated Level Term Period': 'Unknown',
                #  'SOA Guaranteed Level Term Period': ' 5 yr guaranteed',
                #  'SOA Post level term indicator': 'Post Level Term',
                #  'Select_Ultimate_Indicator': 'Select',
                #  'Smoker Status': 'NonSmoker'}


if __name__ == '__main__':
    getYieldData()

    durange = range(40)
    plt.plot(durange, getAnnualYield(durange=durange, intertype='linear'))
    plt.plot(durange, getAnnualYield(durange=durange, intertype='quadratic'))
