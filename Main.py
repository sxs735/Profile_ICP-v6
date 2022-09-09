import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from Model import *

class AppWindow:
    Model_lib = {}
    Data_list = []
    Target_list = []
    Selected = []

    #Default_value
    State = 'Main'
    Sampling = [0.5, 0.5]
    ColorMap = ['Miscellaneous','gist_rainbow','_r']
    ICP_parameter = [True,'PointToPoint',0.1,'None',np.nan]

    # Material
    material_cloud = rendering.MaterialRecord()
    material_cloud.point_size = 12
    material_cloud.shader = 'defaultUnlit'
    material_wire = rendering.MaterialRecord()
    material_wire.base_color = [0, 0, 0, 0]
    material_wire.shader = 'unlitLine'
    material_wire.line_width = 2
    material_mesh = rendering.MaterialRecord()
    material_mesh.base_color = [0.7, 0.7, 0.7, 1]
    material_mesh.shader = 'defaultUnlit'
    material_sphere = rendering.MaterialRecord()
    material_sphere.shader = 'defaultUnlit'

    def __init__(self, width, height):
        self.app = gui.Application.instance
        self.window = self.app.create_window(self.State, width, height)
        self.em = self.window.theme.font_size
        separation_height = int(round(0.5 * self.em))

        self._info = gui.Label('')
        self._info.visible = False
        self.Manual = gui.Label('Manual')
        self.Manual.background_color = gui.Color(0,0,0,0)
        self.Manual.text_color = gui.Color(0,0,0,1)
        self.Manual.visible = True

        #Target
        self.Target_button_load = gui.Button('Load')
        self.Target_button_load.horizontal_padding_em = 0
        self.Target_button_load.vertical_padding_em = 0
        #self.Target_button_load.set_on_clicked(self.Target_Load_clicked)

        self.Target_button_del = gui.Button('Del')
        self.Target_button_del.horizontal_padding_em = 0
        self.Target_button_del.vertical_padding_em = 0
        #self.Target_button_del.set_on_clicked(self.Target_Delete)
        self.Target_button_del.enabled = False

        self.Target_View = gui.ListView()
        self.Target_View.set_max_visible_items(6)
        #self.Target_View.set_on_selection_changed(self.Target_View_mouse)

        self.Change_Tbutton = gui.Button('')
        self.Change_Tbutton.horizontal_padding_em = 0.5
        self.Change_Tbutton.vertical_padding_em = 0
        #self.Change_Tbutton.set_on_clicked(self.Change_Target)
        self.Change_Tbutton.enabled = False

        target_bar = gui.Horiz(0.25 * self.em)
        target_bar.add_child(gui.Label('Target'))
        target_bar.add_child(self.Change_Tbutton)
        target_bar.add_stretch()
        target_bar.add_child(self.Target_button_load)
        target_bar.add_child(self.Target_button_del)

        self.Coeff_name = gui.TextEdit()
        self.Coeff_name.enabled = False
        self.Coeff_button_load = gui.Button('Load')
        self.Coeff_button_load.horizontal_padding_em = 0
        self.Coeff_button_load.vertical_padding_em = 0
        #self.Coeff_button_load.set_on_clicked(self.Coeff_Load_dialog)

        Coeff_bar = gui.Horiz(0.25 * self.em)
        Coeff_bar.add_child(self.Coeff_name)
        Coeff_bar.add_child(self.Coeff_button_load)
        
        #Data
        self.Data_button_load = gui.Button('Load')
        self.Data_button_load.horizontal_padding_em = 0
        self.Data_button_load.vertical_padding_em = 0
        #self.Data_button_load.set_on_clicked(self.Data_Load_clicked)

        self.Data_button_del = gui.Button('Del')
        self.Data_button_del.enabled = False
        self.Data_button_del.horizontal_padding_em = 0
        self.Data_button_del.vertical_padding_em = 0
        #self.Data_button_del.set_on_clicked(self.Data_Delete)

        self.Data_View = gui.ListView()
        self.Data_View.set_max_visible_items(8)
        #self.Data_View.set_on_selection_changed(self.Data_View_mouse)

        self.Change_Dbutton = gui.Button('')
        self.Change_Dbutton.horizontal_padding_em = 0.5
        self.Change_Dbutton.vertical_padding_em = 0
        #self.Change_Dbutton.set_on_clicked(self.Change_Data)
        self.Change_Dbutton.enabled = False

        self.ICP_button = gui.Button('Manual ICP')
        self.ICP_button.horizontal_padding_em = 1.5
        self.ICP_button.vertical_padding_em = 1
        #self.ICP_button.set_on_clicked(self.Manual_ICP)
        self.ICP_button.enabled = False

        data_bar = gui.Horiz(0.25 * self.em)
        data_bar.add_child(gui.Label('Data'))
        data_bar.add_child(self.Change_Dbutton)
        data_bar.add_stretch()
        data_bar.add_child(self.Data_button_load)
        data_bar.add_child(self.Data_button_del)

        #Sag Error
        self.SagError_cal_button = gui.Button('Sag Error Cal.')
        self.SagError_cal_button.enabled = False
        self.SagError_cal_button.horizontal_padding_em = 0
        self.SagError_cal_button.vertical_padding_em = 0
        #self.SagError_cal_button.set_on_clicked(self.SagError_cal)

        SagError_bar = gui.Horiz(0.25 * self.em)
        SagError_bar.add_stretch()
        SagError_bar.add_child(self.SagError_cal_button)

        self.Console = gui.Vert(0, gui.Margins(0.5*self.em, 0.5*self.em, 0.25*self.em, 0.5*self.em))
        self.Console.add_child(gui.Label('Surface parameter'))
        self.Console.add_child(Coeff_bar)
        self.Console.add_fixed(2*separation_height)
        self.Console.add_child(target_bar)
        self.Console.add_child(self.Target_View)
        self.Console.add_fixed(2*separation_height)
        self.Console.add_child(data_bar)
        self.Console.add_child(self.Data_View)
        self.Console.add_fixed(0.5*separation_height)
        self.Console.add_child(SagError_bar)
        self.Console.add_fixed(2*separation_height)
        self.Console.add_child(self.ICP_button)

        #Detail panel
        self.Direction = gui.Combobox()
        self.Direction.add_item('+')
        self.Direction.add_item('-')
        #self.Direction.set_on_selection_changed(self.Apply_enabled)
        self.Scale_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Scale_value.enabled = False
        self.Max_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Max_value.enabled = False
        self.Min_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Min_value.enabled = False
        self.Avg_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Avg_value.enabled = False
        self.Std_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Std_value.enabled = False

        Value_Grid = gui.VGrid(4, 0.1*self.em,gui.Margins(0*self.em, 0.1*self.em, 0.1*self.em, 0.1*self.em))
        Value_Grid.add_child(gui.Label('Z'))
        Value_Grid.add_child(self.Direction)
        Value_Grid.add_child(gui.Label('Scale'))
        Value_Grid.add_child(self.Scale_value)
        Value_Grid.add_child(gui.Label('Max'))
        Value_Grid.add_child(self.Max_value)
        Value_Grid.add_child(gui.Label(' Min'))
        Value_Grid.add_child(self.Min_value)
        Value_Grid.add_child(gui.Label('Avg'))
        Value_Grid.add_child(self.Avg_value)
        Value_Grid.add_child(gui.Label(' Std'))
        Value_Grid.add_child(self.Std_value)

        self.FilterMax_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.FilterMin_value = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.Filter_button = gui.Button('Filter')
        self.Filter_button.enabled = False
        self.Filter_button.toggleable = True
        self.Filter_button.horizontal_padding_em = 0.5
        self.Filter_button.vertical_padding_em = 0
        #self.Filter_button.set_on_clicked(self.Apply_enabled)

        self.Delete_button = gui.Button('Delete')
        self.Delete_button.horizontal_padding_em = 0
        self.Delete_button.vertical_padding_em = 0
        #self.Delete_button.set_on_clicked(self.Delete_mode)

        Filter_set = gui.VGrid(3, 0.1*self.em,gui.Margins(0.5*self.em, 0.1*self.em, 0.5*self.em, 0.1*self.em))
        Filter_set.add_child(self.Delete_button)
        Filter_set.add_child(gui.Label('Less than'))
        Filter_set.add_child(self.FilterMin_value)
        Filter_set.add_child(self.Filter_button)
        Filter_set.add_child(gui.Label('Bigger than'))
        Filter_set.add_child(self.FilterMax_value)

        self.Apply_button = gui.Button('Apply')
        self.Apply_button.enabled = False
        self.Apply_button.horizontal_padding_em = 0
        self.Apply_button.vertical_padding_em = 0
        #self.Apply_button.set_on_clicked(self.Apply_clicked)

        self.Back_button = gui.Button('Back')
        self.Back_button.enabled = True
        self.Back_button.horizontal_padding_em = 0
        self.Back_button.vertical_padding_em = 0
        #self.Back_button.set_on_clicked(self.Back)

        self.Save_button = gui.Button('Save')
        self.Save_button.enabled = True
        self.Save_button.horizontal_padding_em = 0
        self.Save_button.vertical_padding_em = 0
        #self.Save_button.set_on_clicked(self.Save_clicked)

        self.visible_check = gui.Checkbox('')
        #self.visible_check.set_on_checked(self.Change_visible)
        Detail_button_bar = gui.Horiz(0.25 * self.em)
        Detail_button_bar.add_child(self.visible_check)
        Detail_button_bar.add_stretch()
        Detail_button_bar.add_child(self.Apply_button)
        Detail_button_bar.add_fixed(0.5*self.em)
        Detail_button_bar.add_child(self.Back_button)
        Detail_button_bar.add_fixed(0.5*self.em)
        Detail_button_bar.add_child(self.Save_button)

        self.histogram = gui.ImageWidget(o3d.geometry.Image(np.zeros((10*self.em,16*self.em,3),dtype=np.uint8)))
        self.Data_name = gui.TextEdit()
        self.Data_name.enabled = False

        Detail_tile = gui.Horiz(0.25 * self.em)
        Detail_tile.add_child(gui.Label('Name'))
        Detail_tile.add_child(self.Data_name)

        self.Detail = gui.Vert(0, gui.Margins(0.5*self.em, 0.5*self.em, 0.1*self.em, 0.5*self.em))
        self.Detail.add_child(Detail_tile)
        self.Detail.add_fixed(0.5*self.em)
        self.Detail.add_child(Value_Grid)
        self.Detail.add_child(gui.Label('Histogram'))
        self.Detail.add_child(self.histogram)
        self.Detail.add_fixed(0.5*self.em)
        self.Detail.add_child(Filter_set)
        self.Detail.add_fixed(self.em)
        self.Detail.add_child(Detail_button_bar)
        
        self.colorbar = gui.ImageWidget()
        self.colorbar.visible = False

        self.Cancel_button = gui.Button('Cancel')
        self.Cancel_button.enabled = True
        self.Cancel_button.horizontal_padding_em = 0
        self.Cancel_button.vertical_padding_em = 0
        #self.Cancel_button.set_on_clicked(self.Cancel)

        self.Correct_button = gui.Button('Correct')
        self.Correct_button.enabled = True
        self.Correct_button.horizontal_padding_em = 0
        self.Correct_button.vertical_padding_em = 0

        self.Sub_panel = gui.Horiz(0, gui.Margins(0.5*self.em, 0.5*self.em, 0.25*self.em, 0.5*self.em))
        self.Sub_panel.add_stretch()
        self.Sub_panel.add_child(self.Cancel_button)
        self.Sub_panel.add_stretch()
        self.Sub_panel.add_child(self.Correct_button)
        self.Sub_panel.add_stretch()

        #tab
        self.tabs = gui.StackedWidget()
        self.tabs.add_child(self.Console)
        self.tabs.add_child(self.Detail)
        self.tabs.add_child(self.Sub_panel)
        self.Panel = [self.Console,self.Detail,self.Sub_panel]

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        #self._scene.set_on_mouse(self._on_mouse_widget3d)
        #self._scene.set_on_key(self._on_key_widget3d)
        self.window.add_child(self._scene)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.tabs)
        self.window.add_child(self._info)
        self.window.add_child(self.Manual)
        self.window.add_child(self.colorbar)

        self._RectSelect = gui.Label('')
        self._RectSelect.visible = False
        self._RectSelect.background_color = gui.Color(0,0,0,0.2)
        self.window.add_child(self._RectSelect)

        #Menu
        self.Menu = gui.Menu()
        file_menu = gui.Menu()
        file_menu.add_item("csv2xyz", 1)
        Option_menu = gui.Menu()
        Option_menu.add_item('Drawing settings',11)
        Option_menu.add_item('ICP settings',12)
        self.Menu.add_menu("File", file_menu)
        self.Menu.add_menu("Option", Option_menu)
        gui.Application.instance.menubar = self.Menu
        #self.window.set_on_menu_item_activated(1, self.csv2xyz_dialog)
        #self.window.set_on_menu_item_activated(11, self.Draw_dialog)
        #self.window.set_on_menu_item_activated(12, self.ICP_dialog)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        visible_panel = self.Panel[self.tabs.selected_index]
        height = min(r.height,visible_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height)
        self.tabs.frame = gui.Rect(r.get_right() - width, r.y, width,height)
        
        pref = self._info.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self._info.frame = gui.Rect(r.x, r.get_bottom()-pref.height, pref.width, pref.height)
        pref = self.Manual.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.Manual.frame = gui.Rect(r.get_right()-pref.width, r.get_bottom()-pref.height, pref.width, pref.height)
        pref = self.colorbar.calc_preferred_size(layout_context, gui.Widget.Constraints())
        self.colorbar.frame = gui.Rect(r.get_right()-width-pref.width, r.y, pref.width, pref.height)

        self.window.close_dialog()

#%%
if __name__ == '__main__':
    gui.Application.instance.initialize()
    app = AppWindow(1024, 768)
    gui.Application.instance.run()
# %%