# Plate Reader Library
# Load data from plate readers, returning it in a consistent format
#
# Anton Jackson-Smith

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
import datetime as dt
from collections import OrderedDict
from StringIO import StringIO


def load_victor3(filename):
    data = pd.read_excel(filename)

    # Strip leading zeros from well numbers to match the SpectraMax output format
    data['Well'] = data['Well'].str.replace('([A-Z])0?([1-9][0-9]*)', r'\1\2')

    # Get the pairs of times and measurement names
    pairs = zip(data.columns[4::2], data.columns[5::2])

    # Reshape the data into long form, keeping the different times as separate columns for now
    data = pd.melt(
        data, 
        id_vars=data.columns[:4].tolist() + data.columns[4::2].tolist(), 
        var_name='Measurement', 
        value_name='Data')

    # Set up the MeasurementCount field so we can interlace the other measurements
    data['MeasurementCount'] = data['Repeat'] * len(pairs) - 1

    for i, (time, measurement) in enumerate(pairs):
        # Move the correct measurement time into the 'RealTime' column
        # We adjust the 'Time' column to refer to the time that A1 was read, so we
        # can group measurements on the Time column later.
        data.loc[data['Measurement'] == measurement, 'RealTime'] = data.loc[data['Measurement'] == measurement, time]

        # Update the MeasurementCount field to reflect which measurement this was
        # In the end, the filed will be sequential for the measurements taken.
        data.loc[data['Measurement'] == measurement, 'MeasurementCount'] += i

    # Set the Time column to the first time for that measurement (the time that well A1 was read)
    data['Time'] = data.groupby('MeasurementCount')['RealTime'].transform(lambda x: x.iloc[0])


    # Remove the incrementing numbers on the same measurement
    data['Measurement']  = data['Measurement'].str.replace(r'(.*)\.[0-9]+$', r'\1')

    # Ditch extraneous columns
    data = data[['Plate','Repeat','MeasurementCount','Well','Type','Time','RealTime','Measurement','Data']]

    # Fix time formatting
    data['Time'] = pd.to_timedelta(data['Time'].astype(str))
    data['RealTime'] = pd.to_timedelta(data['RealTime'].astype(str))

    return data

def load_spectramax(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    data = OrderedDict([
        ('Time', []),
        ('Wavelength', []),
        ('Plate', []),
        ('ID', []),
        ('Well', []),
        ('Row', []),
        ('Col', []),
        ('Data', [])
    ])

    t0 = None

    for plate in root.findall("./PlateSections"):
        for platesection in plate:
            time = pd.to_datetime(platesection.attrib['ReadTime'])
            if t0 is None:
                t0 = time

            plateName = platesection.attrib['Name']

            waves = list()
            wavelengths = platesection.findall("./InstrumentSettings/WavelengthSettings/Wavelength")
            for wavelength in wavelengths:
                waves.append(int(wavelength.text))

            for wave in platesection.findall("./Wavelengths/Wavelength"):
                wavelength = waves[int(wave.attrib['WavelengthIndex']) - 1]

                for well in wave.findall("./Wells/Well"):
                    data['Time'].append(time - t0)
                    data['Wavelength'].append(wavelength)
                    data['Plate'].append(plateName)
                    data['ID'].append(int(well.attrib['WellID']))
                    data['Well'].append(well.attrib['Name'])
                    data['Row'].append(int(well.attrib['Row']))
                    data['Col'].append(int(well.attrib['Col']))
                    data['Data'].append(np.float(well[0].text))

    return pd.DataFrame(data)


def blank_plate_labels():
    return pd.DataFrame(np.NaN, index=['A','B','C','D','E','F','G','H'], columns=range(1,13), dtype=np.object)


def label(data, labels):
    if not isinstance(labels, dict):
        label_dict = dict()
        for plate in data['Plate'].unique():
            label_dict[plate] = labels
        labels = label_dict

    all_labels = pd.DataFrame(columns=['Row','Column','Label','Well'])
    for key, value in labels.iteritems():
        labels_long = value.stack().rename_axis(["Row","Column"]).rename('Label').reset_index()
        labels_long["Well"] = labels_long.loc[:,'Row':'Column'].apply(lambda x: "{:s}{:g}".format(*x), axis=1)
        labels_long["Plate"] = key
        all_labels = all_labels.append(labels_long)

    return pd.merge(data, all_labels.loc[:,['Plate','Well','Label']].dropna(), on=['Plate', 'Well'])


def show_labels(labels):
    unique_labels = np.unique(labels.values.ravel())
    sns.heatmap(labels.applymap(unique_labels.tolist().index).astype(int), 
            annot=labels.fillna('').apply(lambda x: x.str.extract('(?:.* )*(.*)$')),
            fmt='', cbar=False)


def labels_from_tsv(labels):
    raw_labels = pd.read_table(StringIO(labels), names=range(1,13), dtype=np.str)
    new_labels = blank_plate_labels()
    new_labels.ix[:raw_labels.shape[0]] = raw_labels.values
    return new_labels

def plot(data, labels=None):
    if labels is None:
        labels = data['Label'].unique().tolist()

    fig = plt.figure(figsize=(12,12))
    ax = sns.tsplot(
        data=data.dropna()[data['Label'].isin(labels)], 
        time="Time", 
        condition="Label", 
        unit="Well", 
        value="Data")

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, y: pd.to_timedelta(x)))
    fig.autofmt_xdate()