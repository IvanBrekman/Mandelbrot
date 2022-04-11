#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <cmath>
#include <QDebug>

#include <xmmintrin.h>
#include <pmmintrin.h>

const int HEIGHT    = 600;
const int WIDTH     = HEIGHT * 1.5;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->setFixedSize(WIDTH, HEIGHT);

    frame = new QFrame(this);
    frame->setGeometry(0, 9 * HEIGHT / 20, WIDTH / 10, HEIGHT / 10);
    frame->setFrameShape(QFrame::Box);
    frame->setLineWidth(3);
    frame->setStyleSheet("color: red;");
    mouse_pos = frame->pos();

    this->installEventFilter(this);

    palette = (QColor*) calloc(iterations + 1, sizeof(QColor));
    for (int i = 0; i < iterations; i++) {
        palette[i] = QColor((int) (255 * pow(sin(i / 30.f + 0.5), 2)),
                            (int) (255 * pow(sin(i / 30.f + 1.0), 2)),
                            (int) (255 * pow(sin(i / 30.f + 1.7), 2))
                            );
    }
    palette[iterations] = QColor(0, 0, 0);

    rect.xa = -2;
    rect.ya = -1;
    rect.xb =  1;
    rect.yb =  1;

    history = (ComplexRect*) calloc(hist_max, sizeof(ComplexRect));

    connect(&m_timer, SIGNAL(timeout()), this, SLOT(repaint()));
    m_timer.setInterval(0);
    m_timer.start();

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::paintEvent(QPaintEvent *e)
{
    Q_UNUSED(e);

    ui->dx_dsb->clearFocus();
    ui->dy_dsb->clearFocus();
    ui->dscale_dsb->clearFocus();

    QPainter qp;

    qp.begin(this);

    if      (calculation_type == CalcType::NO_SSE)  this->draw_mand_no_sse  (&qp);
    else if (calculation_type == CalcType::USE_SSE) this->draw_mand_with_sse(&qp);

    qp.end();

    print_fps();
}

void MainWindow::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Z) {
        this->pop_history();

        debug_rect();
        repaint();

        return;
    }

    this->add_history();

    float w  = rect.xb - rect.xa;
    float h  = rect.yb - rect.ya;

    if (e->key() == Qt::Key_D) { rect.xa += w * dx; rect.xb += w * dx; }
    if (e->key() == Qt::Key_A) { rect.xa -= w * dx; rect.xb -= w * dx; }
    if (e->key() == Qt::Key_S) { rect.ya += h * dy; rect.yb += h * dy; }
    if (e->key() == Qt::Key_W) { rect.ya -= h * dy; rect.yb -= h * dy; }

    if (QApplication::keyboardModifiers() & Qt::ControlModifier) {
        float dd = dscale / 2;

        if (e->key() == Qt::Key_Equal) {
            rect.xa += w * dd;
            rect.xb -= w * dd;
            rect.ya += h * dd;
            rect.yb -= h * dd;
        }
        if (e->key() == Qt::Key_Minus) {
            float pd = 1 - dscale;

            rect.xa -= (w / pd) * dd;
            rect.xb += (w / pd) * dd;
            rect.ya -= (h / pd) * dd;
            rect.yb += (h / pd) * dd;
        }
    }

    debug_rect();
    repaint();
}

bool MainWindow::eventFilter(QObject* , QEvent *event)
{
    switch (event->type()) {
        case QEvent::MouseMove: {
            if (! frame->isVisible()) break;

            QMouseEvent *mouse_event = static_cast<QMouseEvent*>(event);
            mouse_pos = mouse_event->pos();
            frame->move(mouse_pos - QPoint(frame->width() / 2, frame->height() / 2));

            break;
        }

        case QEvent::MouseButtonDblClick: {
            if (! frame->isVisible()) break;

            add_history();

            float new_w = (rect.xb - rect.xa) / this->width();
            float new_h = (rect.yb - rect.ya) / this->height();

            rect.xa += frame->x() * new_w;
            rect.ya += frame->y() * new_h;
            rect.xb  = rect.xa + frame->width()  * new_w;
            rect.yb  = rect.ya + frame->height() * new_h;

            frame->setGeometry(0, 9 * HEIGHT / 20, WIDTH / 10, HEIGHT / 10);

            debug_rect();
            repaint();

            break;
        }

        default:
            break;
    }

    return false;
}

void MainWindow::draw_mand_no_sse(QPainter* canvas)
{
    float img_x = this->width();
    float img_y = this->height();

    for (int iy = 0; iy < img_y; iy++) {
        for (int ix = 0; ix < img_x; ix++) {
            float x0 = ix * (rect.xb - rect.xa) / img_x + rect.xa;
            float y0 = iy * (rect.yb - rect.ya) / img_y + rect.ya;

            int n = 0;
            float x = x0, y = y0;

            for (; n < iterations; n++) {
                float x2 = x * x;
                float y2 = y * y;
                float xy = x * y;

                float r2 = x2 + y2;

                if (r2 >= r2_max) break;

                x = x2 - y2 + x0;
                y = xy + xy + y0;
            }

            QPen pen = QPen(palette[n]);

            canvas->setPen(pen);
            canvas->drawPoint(QPoint(ix, iy));
        }
    }
}

void MainWindow::draw_mand_with_sse(QPainter *canvas)
{
    float img_x = this->width();
    float img_y = this->height();

    float w_sh  = (rect.xb - rect.xa) / img_x;
    float h_sh  = (rect.yb - rect.ya) / img_y;

    __m128 _r2_max_ar = _mm_set_ps1(r2_max);
    __m128 _xa_ps     = _mm_set_ps1(rect.xa);
    __m128 _w_sh_ps   = _mm_set_ps1(w_sh);
    __m128 _one_ps    = _mm_set_ps1(1);

    __m128 _3210_w_ps = _mm_set_ps(3, 2, 1, 0);
           _3210_w_ps = _mm_mul_ps(_3210_w_ps, _w_sh_ps);

    for (int iy = 0; iy < img_y; iy++) {
        for (int ix = 0; ix < img_x; ix += 4) {
            __m128 _x0 = _mm_set_ps1(ix * w_sh);
                   _x0 = _mm_add_ps(_x0, _3210_w_ps);
                   _x0 = _mm_add_ps(_x0, _xa_ps);
            __m128 _y0 = _mm_set_ps1(iy * h_sh + rect.ya);

            __m128 _x  = _mm_movehdup_ps(_x0);
            __m128 _y  = _mm_movehdup_ps(_y0);

            __m128i _n  = _mm_set1_epi32(0);
            for (int iter = 0; iter < iterations; iter++) {
                __m128 _x2 = _mm_mul_ps(_x, _x);
                __m128 _y2 = _mm_mul_ps(_y, _y);
                __m128 _xy = _mm_mul_ps(_x, _y);

                __m128 _r2 = _mm_add_ps(_x2, _y2);

                __m128 _cmp = _mm_cmple_ps(_r2, _r2_max_ar);

                int mask = _mm_movemask_ps(_cmp);
                if (!mask) break;

                _n = _mm_add_epi32(_n, _mm_cvtps_epi32(_mm_and_ps(_cmp, _one_ps)));

                _x = _mm_sub_ps(_x2, _y2); _x = _mm_add_ps(_x, _x0);
                _y = _mm_add_ps(_xy, _xy); _y = _mm_add_ps(_y, _y0);
            }

            int* n = (int*) &_n;
            for (int i = 0; i < 4; i++) {
                QPen pen = QPen(palette[(int)n[i]]);

                canvas->setPen(pen);
                canvas->drawPoint(QPoint(ix + i, iy));
            }
        }
    }
}

void MainWindow::add_history()
{
    if (hist_ind == hist_max) {
        for (int i = 0; i < hist_max - 1; i++) {
            ComplexRect tmp = history[i];
            history[i]      = history[i + 1];
            history[i + 1]  = tmp;
        }
        hist_ind--;
    }

    history[hist_ind].xa = rect.xa;
    history[hist_ind].xb = rect.xb;
    history[hist_ind].ya = rect.ya;
    history[hist_ind].yb = rect.yb;

    hist_ind++;
}

void MainWindow::pop_history()
{
    if (hist_ind == 0) return;
    hist_ind--;

    rect.xa = history[hist_ind].xa;
    rect.xb = history[hist_ind].xb;
    rect.ya = history[hist_ind].ya;
    rect.yb = history[hist_ind].yb;
}

void MainWindow::debug_rect()
{
    qDebug() << "Rect: " << rect.xa << " " << rect.ya << " " << rect.xb << " " << rect.yb;
}

void MainWindow::print_fps()
{
    if (frame_count == 0) {
        m_time.start();
    } else if (frame_count == 10) {
        qDebug() << "FPS: " << frame_count / ((float)m_time.elapsed() / 1000.0f);
        frame_count = 0;
        m_time.restart();
    }
    frame_count++;
}

void MainWindow::on_dx_dsb_valueChanged(double arg1)
{
    dx = arg1;
}

void MainWindow::on_dy_dsb_valueChanged(double arg1)
{
    dy = arg1;
}

void MainWindow::on_dscale_dsb_valueChanged(double arg1)
{
    dscale = arg1;
}

void MainWindow::on_show_preferences_cb_stateChanged(int arg1)
{
    ui->preferences->setVisible(arg1);
}

void MainWindow::on_show_frame_cb_stateChanged(int arg1)
{
    frame->setVisible(arg1);
}

void MainWindow::on_use_sse_cb_stateChanged(int arg1)
{
    calculation_type = arg1 ? CalcType::USE_SSE : CalcType::NO_SSE;
    repaint();
}
