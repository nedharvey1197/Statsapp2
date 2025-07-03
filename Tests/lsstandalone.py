import saspy
import pandas as pd
sas = saspy.SASsession()
data = pd.read_csv('temp_data.csv')
sas.df2sd(data, 'testdata')
result = sas.submit("""
options errors=20;
ods output LSMeanDiffCL=work.diffs;
proc glm data=work.testdata;
    class Treatment;
    model TumorSize = Treatment;
    lsmeans Treatment / stderr pdiff cl adjust=bon;
run;
proc print data=work.diffs; run;
""")
print(result['LOG'])
sas.endsas()