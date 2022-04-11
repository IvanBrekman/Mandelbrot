#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QKeyEvent>
#include <QPainter>
#include <QFrame>
#include <QTimer>
#include <QTime>

enum CalcType {
    NO_SSE  = 1,
    USE_SSE = 2
};

struct ComplexRect {
    float xa = 0.f;
    float ya = 0.f;
    float xb = 0.f;
    float yb = 0.f;
};

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow  *ui;
    QColor          *palette;
    QFrame          *frame;
    QPoint           mouse_pos;
    QTimer           m_timer;
    QTime            m_time;

    ComplexRect      rect;
    ComplexRect     *history;

    CalcType         calculation_type = CalcType::NO_SSE;

    int     frame_count = 0;

    int     hist_ind    = 0;
    int     hist_max    = 5000;

    int     iterations  = 255;
    float   r2_max      = 100.f;
    float   dx          = 0.1f;
    float   dy          = 0.1f;
    float   dscale      = 0.1f;

private slots:
    void paintEvent(QPaintEvent* e);
    void keyPressEvent(QKeyEvent* e);
    bool eventFilter(QObject *obj, QEvent *event);

    void draw_mand_no_sse(QPainter* canvas);
    void draw_mand_with_sse(QPainter* canvas);
    void add_history();
    void pop_history();
    void debug_rect();

    void print_fps();

    void on_dx_dsb_valueChanged(double arg1);
    void on_dy_dsb_valueChanged(double arg1);
    void on_dscale_dsb_valueChanged(double arg1);
    void on_show_preferences_cb_stateChanged(int arg1);
    void on_show_frame_cb_stateChanged(int arg1);
    void on_use_sse_cb_stateChanged(int arg1);
};
#endif // MAINWINDOW_H
