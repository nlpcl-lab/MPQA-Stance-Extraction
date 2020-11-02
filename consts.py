NONE = 'O'
PAD = "[PAD]"
UNK = "[UNK]"

# for BERT
CLS = '[CLS]'
SEP = '[SEP]'


ATTITUDES = ['other-attitude',
             'arguing-pos',
             'sentiment-neg',
             'speculation',
             'agree-neg',
             'sentiment-pos',
             'arguing-neg',
             'intention-pos',
             'agree-pos',
             'intention-neg',
             'other']

"""
    28 argument roles
    
    There are 35 roles in ACE2005 dataset, but the time-related 8 roles were replaced by 'Time' as the previous work (Yang et al., 2016).
    ['Time-At-End','Time-Before','Time-At-Beginning','Time-Ending', 'Time-Holds', 'Time-After','Time-Starting', 'Time-Within'] --> 'Time'.
"""
ARGUMENTS = ['Place',
             'Crime',
             'Prosecutor',
             'Sentence',
             'Org',
             'Seller',
             'Entity',
             'Agent',
             'Recipient',
             'Target',
             'Defendant',
             'Plaintiff',
             'Origin',
             'Artifact',
             'Giver',
             'Position',
             'Instrument',
             'Money',
             'Destination',
             'Buyer',
             'Beneficiary',
             'Attacker',
             'Adjudicator',
             'Person',
             'Victim',
             'Price',
             'Vehicle',
             'Time']

# 54 entities
ENTITIES = ['VEH:Water',
            'GPE:Nation',
            'ORG:Commercial',
            'GPE:State-or-Province',
            'Contact-Info:E-Mail',
            'Crime',
            'ORG:Non-Governmental',
            'Contact-Info:URL',
            'Sentence',
            'ORG:Religious',
            'VEH:Underspecified',
            'WEA:Projectile',
            'FAC:Building-Grounds',
            'PER:Group',
            'WEA:Exploding',
            'WEA:Biological',
            'Contact-Info:Phone-Number',
            'WEA:Chemical',
            'LOC:Land-Region-Natural',
            'WEA:Nuclear',
            'LOC:Region-General',
            'PER:Individual',
            'WEA:Sharp',
            'ORG:Sports',
            'ORG:Government',
            'ORG:Media',
            'LOC:Address',
            'WEA:Shooting',
            'LOC:Water-Body',
            'LOC:Boundary',
            'GPE:Population-Center',
            'GPE:Special',
            'LOC:Celestial',
            'FAC:Subarea-Facility',
            'PER:Indeterminate',
            'VEH:Subarea-Vehicle',
            'WEA:Blunt',
            'VEH:Land',
            'TIM:time',
            'Numeric:Money',
            'FAC:Airport',
            'GPE:GPE-Cluster',
            'ORG:Educational',
            'Job-Title',
            'GPE:County-or-District',
            'ORG:Entertainment',
            'Numeric:Percent',
            'LOC:Region-International',
            'WEA:Underspecified',
            'VEH:Air',
            'FAC:Path',
            'ORG:Medical-Science',
            'FAC:Plant',
            'GPE:Continent']

