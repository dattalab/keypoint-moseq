import pytest
import time
from unittest.mock import Mock
from keypoint_moseq.view.jupyter_display import _set_group_labels_widget
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

@pytest.fixture
def initial_labels():
    return {'mouse0': '', 'mouse1': '', 'mouse2': ''}

def test_set_group_labels_widget_mock(initial_labels, dash_duo):
    def edit_and_assert_cell(row_idx, value, expected_values):
        """
        Edit a cell in the Dash DataTable and assert the expected values.

        Data table assumed to have the id group-labels-table

        Parameters
        ----------
        row_idx : int
            1-based index of the row to edit (excluding header).
        value : str
            The value to enter into the cell.
        expected_values : list of str
            The expected values for the group-label column after editing.
        """
        first_col = dash_duo.find_element(f'#group-labels-table tbody tr:nth-child({row_idx+1}) td:nth-child(1)')
        cell = dash_duo.find_element(f'#group-labels-table tbody tr:nth-child({row_idx+1}) td:nth-child(2)')
        cell.click()
        cell_input = cell.find_element_by_css_selector('input.dash-cell-value')
        cell_input.clear()
        cell_input.send_keys(value)
        first_col.click()
        cell_value_divs = dash_duo.find_elements('#group-labels-table tbody tr td div.dash-cell-value')
        cell_text_values = [div.text for div in cell_value_divs][1::2]
        assert cell_text_values == expected_values

    save_group_labels_mock = Mock()
    widget = _set_group_labels_widget(initial_labels, save_group_labels_mock)

    dash_duo.start_server(widget)
    dash_duo.wait_for_text_to_equal('#user-message', 'Click to save group assignments')

    # Test that the table initialized properly
    rows = dash_duo.find_elements('#group-labels-table tbody tr')
    assert [r.text for r in rows[1:]] == ['mouse0', 'mouse1', 'mouse2']
    save_group_labels_mock.assert_not_called()

    # Test editing the table
    edit_and_assert_cell(1, 'WT', ['WT', '', ''])
    edit_and_assert_cell(2, 'HET', ['WT', 'HET', ''])
    edit_and_assert_cell(3, 'WT', ['WT', 'HET', 'WT'])
    edit_and_assert_cell(1, 'HET', ['HET', 'HET', 'WT'])

    # Save callback should not have been called up to this point
    save_group_labels_mock.assert_not_called()

    save_button = dash_duo.find_element('#save-button')
    save_button.click()
    time.sleep(0.5)

    # Now the save callback should have been called with the current state of the table
    save_group_labels_mock.assert_called_once_with({
        'mouse0': 'HET',
        'mouse1': 'HET',
        'mouse2': 'WT',
    })
    
    first_cell = dash_duo.find_element('#group-labels-table tbody tr:nth-child(2) td:nth-child(2)')
    second_cell = dash_duo.find_element('#group-labels-table tbody tr:nth-child(3) td:nth-child(2)')
    third_cell = dash_duo.find_element('#group-labels-table tbody tr:nth-child(4) td:nth-child(2)')
    first_col = dash_duo.find_element(f'#group-labels-table tbody tr:nth-child(2) td:nth-child(1)')

    rows = dash_duo.find_elements('#group-labels-table tbody tr')
    print([row.text for row in rows])

    dash_duo.driver.save_screenshot('tests/selenium_screenshots/test_set_group_labels_widget_mock/before_copy_paste.png')

    # 1. Shift-select cells 1 and 2
    # 2. ctrl+c to copy
    # 3. click to select cell 3
    # 4. ctrl+v to paste, introducing erroneous extra rows to the data table
    # (
    #     ActionChains(dash_duo.driver)
    #     .click(first_cell)
    #     .key_down(Keys.SHIFT)
    #     .click(second_cell)
    #     .key_up(Keys.SHIFT)
    #     .key_down(Keys.CONTROL)
    #     .send_keys('c')
    #     .key_up(Keys.CONTROL)
    #     .click(third_cell)
    #     .key_down(Keys.CONTROL)
    #     .send_keys('v')
    #     .key_up(Keys.CONTROL)
    #     .click(first_col)
    #     .perform()
    # )
    (
        ActionChains(dash_duo.driver)
        .click(first_cell)
        .key_down(Keys.SHIFT)
        .click(second_cell)
        .key_up(Keys.SHIFT)
        .perform()
    )

    dash_duo.driver.save_screenshot('tests/selenium_screenshots/test_set_group_labels_widget_mock/first_two_selected.png')

    # Check which element has focus
    focused_element = dash_duo.driver.execute_script("return document.activeElement")
    print(f"Focused element before copy: {focused_element.get_attribute('outerHTML')}")

    (
        ActionChains(dash_duo.driver)
        .send_keys(Keys.CONTROL + 'c')
        .click(third_cell)
        .perform()
    )

    dash_duo.driver.save_screenshot('tests/selenium_screenshots/test_set_group_labels_widget_mock/after_third_cell_click.png')

    third_cell_input = third_cell.find_element_by_css_selector('input.dash-cell-value')
    third_cell_input.clear()
    third_cell_input.send_keys(Keys.CONTROL + 'v')
    first_col.click()

    dash_duo.driver.save_screenshot('tests/selenium_screenshots/test_set_group_labels_widget_mock/after_paste.png')

    # (
    #     ActionChains(dash_duo.driver)
    #     .send_keys(Keys.CONTROL + 'v')
    #     .click(first_col)
    #     .perform()
    # )

    rows = dash_duo.find_elements('#group-labels-table tbody tr')
    # There should be an erroneous extra row now, plus the header row
    print([row.text for row in rows])
    assert len(rows) == 5

    save_button.click()
    time.sleep(0.5)

    # But the extra row should have been filtered out before the Controller callback is called
    # and the third cell should have been updated by the paste operation
    second_call_args = save_group_labels_mock.call_args_list[-1][0][0]
    assert second_call_args == {
        'mouse0': 'HET',
        'mouse1': 'HET',
        'mouse2': 'HET'
    }