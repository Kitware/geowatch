

def main():
    # import matplotlib.pyplot as plt
    import dataframe_image as dfi
    import pandas as pd
    df = pd.DataFrame(
        {
            'a': [1, 2, 3],
            'b': [1, 2, 3],
        }
    )

    def highlight_df(df):

        df_style = df.copy().astype(str)
        df_style.loc[:, :] = None

        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                if str(df.iloc[r][c]) != "2" and r != c:
                    df_style.iat[r, c] = "background-color: bisque"

        return df_style

    df_styled = df.style.apply(highlight_df, axis=None)
    df_styled.set_caption('My Caption')

    dfi_converter = "chrome"  # matplotlib

    dfi.export(
        df_styled,
        'foo.png',
        table_conversion=dfi_converter,
        fontsize=12,
        max_rows=-1,
    )


    import dataframe_image as dfi
    import pandas as pd
    df = pd.DataFrame(
        {
            'a': [1, 2, 3],
            'b': [1, 2, 3],
        }
    )

    df_style = df.style.set_caption('My Caption')
    dfi_converter = "chrome"  # matplotlib
    dfi.export(
        df_style,
        'bar.png',
        table_conversion=dfi_converter,
        fontsize=12,
        max_rows=-1,
    )
