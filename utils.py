import pandas as pd


def parse_date(s: str) -> str:
    """
    Parseia datas no formato mmm/yy para formato ISO

    Parameters
    ----------

    s: str
        string contendo data no formato mmm/yy

    Returns
    -------
    str
        string no formato ISO de data (yyyy-mm-dd)
    """
    month_dict = {
        'jan': 1,
        'fev': 2,
        'mar': 3,
        'abr': 4,
        'mai': 5,
        'jun': 6,
        'jul': 7,
        'ago': 8,
        'set': 9,
        'out': 10,
        'nov': 11,
        'dez': 12,
    }

    month, year = s.split("/")
    month = month_dict[month]

    return f"20{year}-{month:02d}-01"


def transform_dataframe(original_df: pd.DataFrame, f) -> pd.DataFrame:
    """
    Cria novo DataFrame com valores do 'original_dataframe' transformados
    pela função 'f'

    Parameters
    ----------

    original_df: pd.DataFrame
        DataFrame com os valores originais

    f: Callable
        Função que realiza transformação. Deve ter "signature" f(df, i, col_name)

    Returns
    -------
    pd.DataFrame
        Novo DataFrame com valores transformados
    """
    # Cria dicionário de listas vazias com nomes de colunas
    delta_dict = {}
    for col_name, _ in original_df.items():
        if col_name == "month":
            continue
        delta_dict[col_name] = [0] # Coloca zeros na primeira linha

    # Popula novo DataFrame com a diferença dos dados no primeiro
    for i, row in original_df.iterrows():
        # Pula primeira linha
        if i == 0:
            continue
        for col_name, _ in row.items():
            # Ignora coluna de meses
            if col_name == "month":
                continue
            # Salva diferenças em dicionário
            delta_dict[col_name].append(f(original_df, i, col_name))

    return pd.DataFrame(delta_dict)


def save_tranformed_df(filepath: str, df: pd.DataFrame, f) -> None:
    """
    Salva arquivo csv com valores de 'df' transformados segundo função 'f'

    Parameters
    ----------
    filepath: str
        Path onde salvar o arquivo novo

    df: pd.DataFrame
        DataFrame com os valores originais

    f: Callable
        Função que realiza transformação. Deve ter "signature" f(df, i, col_name)

    Returns
    -------
    None
    """

    delta_df = transform_dataframe(df, f)

    # Usa coluna de meses como índice
    tmp = df['month']
    delta_df['month'] = tmp
    # Remove primeira linha, pois calcula a diferença, então não faria sentido ter
    # o primeiro valor
    delta_df = delta_df.drop([0])
    delta_df = delta_df.set_index('month')

    delta_df.to_csv(filepath, sep=";", decimal=",", index_label='month')

