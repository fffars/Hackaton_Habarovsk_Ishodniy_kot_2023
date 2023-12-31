/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/GUIForms/JFrame.java to edit this template
 */
package test_application2;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtils;
import org.jfree.chart.JFreeChart;
import org.jfree.data.statistics.HistogramDataset;
//import org.apache.commons.io.FilenameUtils;


import java.io.File;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.charset.CharsetDecoder;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.ImageIcon;
import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

/**
 *
 * @author Евгений
 */
public class Test_application extends javax.swing.JFrame {
    
    String filePath;
    StringBuilder string = new StringBuilder();
    Process p;

    /**
     * Creates new form NewJFrame
     */
    public Test_application() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        jLabel1 = new javax.swing.JLabel();
        jButton1 = new javax.swing.JButton();
        jButton3 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jLabel2 = new javax.swing.JLabel();
        jScrollPane1 = new javax.swing.JScrollPane();
        jTextPane1 = new javax.swing.JTextPane();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Test_Application");
        setBackground(new java.awt.Color(21, 58, 87));
        setForeground(new java.awt.Color(255, 51, 51));
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosed(java.awt.event.WindowEvent evt) {
                formWindowClosed(evt);
            }
        });

        jPanel1.setBackground(new java.awt.Color(21, 58, 87));

        jLabel1.setIcon(new javax.swing.ImageIcon(getClass().getResource("/resources/2023-09-22_19-44-30.png"))); // NOI18N
        jLabel1.setName(""); // NOI18N

        jButton1.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        jButton1.setForeground(new java.awt.Color(21, 100, 192));
        jButton1.setText("Открыть файл");
        jButton1.setBorder(null);
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jButton3.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        jButton3.setForeground(new java.awt.Color(21, 100, 192));
        jButton3.setText("Сохранить CSV файл");
        jButton3.setBorder(null);

        jButton2.setFont(new java.awt.Font("Segoe UI", 0, 14)); // NOI18N
        jButton2.setForeground(new java.awt.Color(21, 100, 192));
        jButton2.setText("Запустить");
        jButton2.setBorder(null);
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        jScrollPane1.setViewportView(jTextPane1);

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 550, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jLabel2, javax.swing.GroupLayout.DEFAULT_SIZE, 6, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                        .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 396, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                            .addComponent(jButton1, javax.swing.GroupLayout.PREFERRED_SIZE, 192, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                            .addComponent(jButton2, javax.swing.GroupLayout.PREFERRED_SIZE, 198, javax.swing.GroupLayout.PREFERRED_SIZE)))
                    .addComponent(jButton3, javax.swing.GroupLayout.PREFERRED_SIZE, 192, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap())
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jScrollPane1)
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jLabel1, javax.swing.GroupLayout.PREFERRED_SIZE, 433, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(jLabel2, javax.swing.GroupLayout.PREFERRED_SIZE, 404, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(jButton1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addComponent(jButton2, javax.swing.GroupLayout.PREFERRED_SIZE, 63, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jButton3, javax.swing.GroupLayout.PREFERRED_SIZE, 63, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(0, 0, Short.MAX_VALUE)))
                .addContainerGap())
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        final JFileChooser fc = new JFileChooser();
        fc.addChoosableFileFilter(new MyFilter());

        int returnVal = fc.showOpenDialog(this);

        if (returnVal == JFileChooser.APPROVE_OPTION) {
            File file = fc.getSelectedFile();
            filePath = file.getAbsolutePath();
            //This is where a real application would open the file.
//            log.append("Opening: " + file.getName() + "." + newline);
        } else {
//            log.append("Open command cancelled by user." + newline);
        }
    }//GEN-LAST:event_jButton1ActionPerformed

    public CharBuffer getCharBuffer(byte[] str)
    {
        Charset charset = Charset.forName("CP1251");
CharsetDecoder decoder = charset.newDecoder();
//ByteBuffer.wrap simply wraps the byte array, it does not allocate new memory for it
ByteBuffer srcBuffer = ByteBuffer.wrap(str);
        try {
            //Now, we decode our srcBuffer into a new CharBuffer (yes, new memory allocated here, no can do)
            CharBuffer resBuffer = decoder.decode(srcBuffer);
            return resBuffer;
        } catch (CharacterCodingException ex) {
            Logger.getLogger(Test_application.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }
    
    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
//        ProcessBuilder process = new ProcessBuilder("python", "script.py", "123");
        byte[] str = {1};
//        string.
//        string.append("Имя файла: ");
//        jTextPane1.setText(string.toString());
        
        
        try {
            ProcessBuilder pb = new ProcessBuilder();
            p = pb.command(".venv\\Scripts\\activate.bat\\").start();
            p.waitFor();
            p = pb.command("streamlit", "run", "script.py").start();
            // "streamlit", "run", "script.py"
        
        // receive from child
        new Thread(() -> {
            try {
                int c;
                while ((c = p.getInputStream().read(str)) != -1)
//                    str[0] = (byte)c;
                    string.append(getCharBuffer(str));
//                    System.out.println(getCharBuffer(str));
                    jTextPane1.setText(string.toString());
//                    System.out.write((byte)c);
                
//                    jTextPane1
            } catch (Exception e) {
                e.printStackTrace();
            }
            try {
                int c;
                while ((c = p.getErrorStream().read(str)) != -1)
//                    str[0] = (byte)c;
                    string.append(getCharBuffer(str));
//                    System.out.println(getCharBuffer(str));
                    jTextPane1.setText(string.toString());
//                    jTextPane1
            } catch (Exception e) {
                e.printStackTrace();
            }
        }).start();
        // send to child
        try (Writer w = new OutputStreamWriter(p.getOutputStream(), "UTF-8")) {
            w.write("send to child\n");
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(Test_application.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(Test_application.class.getName()).log(Level.SEVERE, null, ex);
        }
//        try {
//            p.waitFor();
//        } catch (InterruptedException ex) {
//            Logger.getLogger(Test_application.class.getName()).log(Level.SEVERE, null, ex);
//        }
//        try {
////            Process start = p.start();
////            start.getOutputStream();
//        } catch (IOException ex) {
//            Logger.getLogger(Test_application.class.getName()).log(Level.SEVERE, null, ex);
//        }
        } catch (IOException ex) {
            Logger.getLogger(Test_application.class.getName()).log(Level.SEVERE, null, ex);
        } catch (InterruptedException ex) {
            Logger.getLogger(Test_application.class.getName()).log(Level.SEVERE, null, ex);
        }
//        jLabel2.setIcon(new ImageIcon("histogram.png"));
    }//GEN-LAST:event_jButton2ActionPerformed

    private void formWindowClosed(java.awt.event.WindowEvent evt) {//GEN-FIRST:event_formWindowClosed
        p.destroy();
    }//GEN-LAST:event_formWindowClosed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) throws IOException {
//        double[] vals = {
//
//                0.71477137, 0.55749811, 0.50809619, 0.47027228, 0.25281568,
//                0.66633175, 0.50676332, 0.6007552, 0.56892904, 0.49553407,
//                0.61093935, 0.65057417, 0.40095626, 0.45969447, 0.51087888,
//                0.52894806, 0.49397198, 0.4267163, 0.54091298, 0.34545257,
//                0.58548892, 0.3137885, 0.63521146, 0.57541744, 0.59862265,
//                0.66261386, 0.56744017, 0.42548488, 0.40841345, 0.47393027,
//                0.60882106, 0.45961208, 0.43371424, 0.40876484, 0.64367337,
//                0.54092033, 0.34240811, 0.44048106, 0.48874236, 0.68300902,
//                0.33563968, 0.58328107, 0.58054283, 0.64710522, 0.37801285,
//                0.36748982, 0.44386445, 0.47245989, 0.297599, 0.50295541,
//                0.39785732, 0.51370486, 0.46650358, 0.5623638, 0.4446957,
//                0.52949791, 0.54611411, 0.41020067, 0.61644868, 0.47493691,
//                0.50611458, 0.42518211, 0.45467712, 0.52438467, 0.724529,
//                0.59749142, 0.45940223, 0.53099928, 0.65159718, 0.38038268,
//                0.51639554, 0.41847437, 0.46022878, 0.57326103, 0.44913632,
//                0.61043611, 0.42694949, 0.43997814, 0.58787928, 0.36252603,
//                0.50937634, 0.47444256, 0.57992527, 0.29381335, 0.50357977,
//                0.42469464, 0.53049697, 0.7163579, 0.39741694, 0.41980533,
//                0.68091159, 0.69330702, 0.50518926, 0.55884098, 0.48618324,
//                0.48469854, 0.55342267, 0.67159111, 0.62352006, 0.34773486};
//
//
//        HistogramDataset dataset = new HistogramDataset();
//        dataset.addSeries("Значения", vals, 50);
//
//        JFreeChart histogram = ChartFactory.createHistogram("Нормальное распределение",
//                "X", "Y", dataset);
//
//        ChartUtils.saveChartAsPNG(new File("histogram.png"), histogram, 450, 400);
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Windows".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Test_application.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Test_application.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Test_application.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Test_application.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new Test_application().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JTextPane jTextPane1;
    // End of variables declaration//GEN-END:variables
}

class MyFilter extends FileFilter {

    @Override
    public boolean accept(File f) {
        if (f.isDirectory()) {
            return true;
        }

// String extension = FilenameUtils.getExtension(f.getName());
// if (extension != null) {
        if (f.getName().substring(f.getName().lastIndexOf(".") + 1).equals("mp4")) {
            return true;
        } else {
            return false;
        }
    }

    @Override
    public String getDescription() {
        return "Видео в формате mp4";
    }
}
