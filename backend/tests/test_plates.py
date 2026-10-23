import numpy as np

from anpr.batch import interpolate
from anpr.plates import PlateVotes, format_license, get_car, license_complies_format


def test_plate_format():
    assert license_complies_format('AB12CDE')
    assert license_complies_format('4BI2CDE')  # OCR mix-ups at the right positions are fixable
    assert not license_complies_format('AB12CD')
    assert not license_complies_format('ABC2CDE')
    assert format_license('4BI2CD5') == 'AB12CDS'


def test_get_car():
    tracks = np.array([[0, 0, 10, 10, 1], [20, 20, 40, 40, 2]])
    assert get_car((25, 25, 30, 30), tracks)[4] == 2
    assert get_car((5, 5, 15, 15), tracks) is None
    assert get_car((32, 25, 44, 30), tracks)[4] == 2  # plate sticking out of the car box still counts


def test_votes():
    votes = PlateVotes()
    for text, score in [('NA13NRU', 0.5), ('NA13HRU', 0.4), ('MA13NRU', 0.85), ('NA13NRU', 0.6)]:
        votes.add(text, score)
    assert votes.text() == 'NA13NRU'  # a single high-scoring misread doesn't win
    assert votes.reads == 4 and 0 < votes.score() < 1


def test_interpolate_fills_gaps():
    row = lambda f, x: {'frame_nmr': str(f), 'car_id': '1', 'car_bbox': f'[{x} 0.0 {x} 0.0]',
                        'license_plate_bbox': '[0.0 0.0 0.0 0.0]', 'license_plate_bbox_score': '0.9',
                        'license_number': 'AB12CDE', 'license_number_score': '0.8'}
    rows = interpolate([row(0, 0.0), row(4, 8.0)])
    assert [int(r['frame_nmr']) for r in rows] == [0, 1, 2, 3, 4]
    assert rows[2]['car_bbox'] == '[4.0 0.0 4.0 0.0]' and rows[2]['license_number'] == '0'


if __name__ == '__main__':
    test_plate_format()
    test_get_car()
    test_votes()
    test_interpolate_fills_gaps()
    print('ok')
