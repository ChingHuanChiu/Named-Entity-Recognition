from enum import Enum

HUGGINGFACE_MODEL = "hfl/chinese-roberta-wwm-ext"
SEQ_MAX_LENGTH = 500

class B(Enum):
    _2 = "B_Thing"
    _4 = "B_Person"
    _6 = "B_Location"
    _8 = "B_Time"
    _10 = "B_Metric"
    _12 = "B_Organization"
    _14 = "B_Abstract"
    _16 = "B_Physical"
    _18 = "B_Term"

class I(Enum):
    _3 = "I_Thing"
    _5 = "I_Person"
    _7 = "I_Location"
    _9 = "I_Time"
    _11 = "I_Metric"
    _13 = "I_Organization"
    _15 = "I_Abstract"
    _17 = "I_Physical"
    _19 = "I_Term"

class O(Enum):
    _1 = "O"



class SpecialToken(Enum):
    _0 = "[PAD]"
    _20 = "[SEP]"
