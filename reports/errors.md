Traceback (most recent call last):
  File "/home/user228/python_labs/Fish-Dimensions-Regression-Analysis/run.py", line 12, in <module>
    navbar = dbc.NavbarSimple(
             ^^^^^^^^^^^^^^^^^
  File "/home/user228/python_labs/Fish-Dimensions-Regression-Analysis/.venv/lib/python3.12/site-packages/dash/development/base_component.py", line 504, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/user228/python_labs/Fish-Dimensions-Regression-Analysis/.venv/lib/python3.12/site-packages/dash_bootstrap_components/_components/NavbarSimple.py", line 135, in __init__
    super(NavbarSimple, self).__init__(children=children, **args)
  File "/home/user228/python_labs/Fish-Dimensions-Regression-Analysis/.venv/lib/python3.12/site-packages/dash/development/base_component.py", line 204, in __init__
    raise TypeError(
TypeError: The `dash_bootstrap_components.NavbarSimple` component (version 2.0.4) received an unexpected keyword argument: `light`
Allowed arguments: brand, brand_external_link, brand_href, brand_style, children, className, class_name, color, dark, expand, fixed, fluid, id, key, links_left, sticky, style



Traceback (most recent call last):
  File "/home/user228/python_labs/Fish-Dimensions-Regression-Analysis/run.py", line 93, in <module>
    app.run_server(debug=True)
    ^^^^^^^^^^^^^^
  File "/home/user228/python_labs/Fish-Dimensions-Regression-Analysis/.venv/lib/python3.12/site-packages/dash/_obsolete.py", line 22, in __getattr__
    raise err.exc(err.message)
dash.exceptions.ObsoleteAttributeException: app.run_server has been replaced by app.run