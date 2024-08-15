;; Added by Package.el.  This must come before configurations of
;; installed packages.  Don't delete this line.  If you don't want it,
;; just comment it out by adding a semicolon to the start of the line.
;; You may delete these explanatory comments.
(package-initialize)

;;;------------------------------------------------------------------------;;;
;;; MELPA packages
;;;------------------------------------------------------------------------;;;
;; (add-to-list 'package-archives '("melpa" . "http://melpa.org/packages/") t)
;; (add-to-list 'load-path "~/.emacs.d/elisp/")
;; (require 'package)
;; (require 'use-package)
(require 'cmake-mode)
;; (require 'modern-cpp-font-lock)
;; (require 'ccls)

;; (use-package modern-cpp-font-lock
;;   :ensure t
;;   :hook (c++-mode . modern-c++-font-lock-mode))

;; (use-package lsp-mode :commands lsp)

;; (use-package ccls
;; :hook ((c-mode c++-mode objc-mode cuda-mode) .
;;        (lambda () (require 'ccls) (lsp))))


;;;------------------------------------------------------------------------;;;
;;; Set modes and interface stuff                                          ;;;
;;;------------------------------------------------------------------------;;;
(delete-selection-mode t)               ; I like it... so sue me
(global-font-lock-mode t)               ; enable syntax highlighting
(column-number-mode t)                  ; enable column numbers

(menu-bar-mode -1)                      ; no menus
(blink-cursor-mode nil)                 ; or blinking

(setq inhibit-startup-message t)        ; don't show startup screen
(setq display-time-day-and-date nil)    ; don't show the date and time
(setq make-backup-files nil)            ; never actually used them
(setq auto-save-default nil)
(setq ring-bell-function                ; instead of beeping
      (lambda () (message "pip...")))           

(fset 'yes-or-no-p 'y-or-n-p)           ; yes/no questions become y/n

;;;------------------------------------------------------------------------;;;
;;; Set colors                                                             ;;;
;;;------------------------------------------------------------------------;;;
(load-theme 'wombat t)

;;;------------------------------------------------------------------------;;;
;;; Set defaults                                                           ;;;
;;;------------------------------------------------------------------------;;;
(setq-default major-mode 'text-mode)
;; (setq-default fill-column 96)
(setq-default fill-column 80)
(setq-default comment-fill-column 80)
(setq-default tab-width 4)
(setq-default indent-tabs-mode nil)
(setq-default comment-column 48)
(setq-default truncate-partial-width-windows t) ; word wrap
(setq-default scroll-step 1)                    ; smooth scrolling
(setq-default auto-fill-function 'do-auto-fill) ; turn on auto-fill-mode

(setq fill-column 96)
(setq fill-column 80)
(setq comment-fill-column 80)
(setq tab-width 4)
(setq indent-tabs-mode nil)
(setq comment-column 48)
(setq truncate-partial-width-windows t)
(setq scroll-step 1)
(setq auto-fill-function 'do-auto-fill)

;;;------------------------------------------------------------------------;;;
;;; C programming defaults                                                 ;;;
;;;------------------------------------------------------------------------;;;
(setq-default c-basic-indent 4)
(setq-default c-basic-offset 4)
(setq-default c-tab-always-indent nil)

(setq c-basic-indent 4)
(setq c-basic-offset 4)
(setq c-tab-always-indent nil)
;; (setq c-default-style "linux")

;;;------------------------------------------------------------------------;;;
;;; Set keybindings                                                        ;;;
;;;------------------------------------------------------------------------;;;
(global-set-key "\r"          'newline-and-indent)
(global-set-key [?\M-g]       'goto-line)
(global-set-key [?\C-c ?\C-c] 'comment-region)
(global-set-key [?\C-c ?\C-v] 'uncomment-region)
(global-set-key [?\M-p]       'fill-paragraph)
(global-set-key [?\M-s]       'fill-sentence)
(global-set-key [?\C-c \t]    'indent-for-comment) 
(global-set-key (kbd "C-x 9") (lambda () (interactive) ; sensible split 
                                (split-window-right 86)))
;;;------------------------------------------------------------------------;;;
;;; Latex stuff                                                            ;;;
;;;------------------------------------------------------------------------;;;
(add-hook 'LaTeX-mode-hook
          (lambda ()
            (setq whitespace-style '(tabs tab-mark))
            (flyspell-mode 1)
            (setq fill-column 72)
            (whitespace-mode 1)))

;;;------------------------------------------------------------------------;;;
;;; Removes all tabs from a line                                           ;;;
;;;------------------------------------------------------------------------;;;
(defun custom-untabify ()
  (save-excursion
    (goto-char (point-min))
    (while (re-search-forward "[ \t]+$" nil t)
      (delete-region (match-beginning 0) (match-end 0)))
    (goto-char (point-min))
    (if (search-forward "\t" nil t)
        (untabify (1- (point)) (point-max))))
  nil)

;;;------------------------------------------------------------------------;;;
;;; LLVM coding style guidelines in emacs                                  ;;;
;;; Maintainer: LLVM Team, http://llvm.org/                                ;;;
;;;------------------------------------------------------------------------;;;
(defun llvm-lineup-statement (langelem)
  (let ((in-assign (c-lineup-assignments langelem)))
    (if (not in-assign)
        '++
      (aset in-assign 0
            (+ (aref in-assign 0)
               (* 2 c-basic-offset)))
      in-assign)))

;;;------------------------------------------------------------------------;;;
;;; C++20 keywords                                                         ;;;
;;;------------------------------------------------------------------------;;;
;; (font-lock-add-keywords 'c++-mode
;;                         '(("co_yield"  . font-lock-keyword-face)
;;                           ("co_await"  . font-lock-keyword-face)
;;                           ("co_return" . font-lock-keyword-face)
;;                           ("consteval" . font-lock-keyword-face)
;;                           ("constinit" . font-lock-keyword-face)
;;                           ("concept"   . font-lock-keyword-face)
;;                           ("requires"  . font-lock-keyword-face)))

;;;------------------------------------------------------------------------;;;
;;; C coding stuff: indentation, deletion, tabification                    ;;;
;;;------------------------------------------------------------------------;;;
(add-hook 'c-mode-common-hook
          (function
           (lambda ()
             (setq c-offsets-alist
                   (append
                    '(
                      ;; (inextern-lang         . 0)
                      ;; (arglist-cont-nonempty . 4)
                      ;; (arglist-close . c-lineup-arglist-close-under-paren)
                      (case-label            . *)  ;
                      (statement-case-intro  . *)  ;
                      (substatement-open     . 0)  ; no indent for empty {
                      (statement-cont        . c-lineup-assignments) ; lines up =
                      (stream-op             . c-lineup-streamop)
                      (label                 . *)  ; labels are shifted 1/2 step
                      (access-label          . /)  ;
                      (brace-list-intro      . +)  ; for array initializers
                      (inlambda              . 0)  ; lambdas look like functions
                      (innamespace           . +)  ; from llvm
                      (arglist-intro         . ++)
                      (arglist-close         . 0)
                      (member-init-intro     . ++) ; from llvm
                      ;; (statement-cont        . llvm-lineup-statement) ; from llvm
                      )
                    c-offsets-alist))
             (c-toggle-hungry-state 0)
             (flyspell-prog-mode)
             (make-local-variable 'write-contents-hooks)
             (add-hook 'write-contents-hooks 'custom-untabify))))
;; (custom-set-variables
;;  ;; custom-set-variables was added by Custom.
;;  ;; If you edit it by hand, you could mess it up, so be careful.
;;  ;; Your init file should contain only one such instance.
;;  ;; If there is more than one, they won't work right.
;;  '(c-noise-macro-names (quote ("constexpr")))
;;  '(custom-safe-themes
;;    (quote
;;     ("dd4db38519d2ad7eb9e2f30bc03fba61a7af49a185edfd44e020aa5345e3dca7" "68769179097d800e415631967544f8b2001dae07972939446e21438b1010748c" default)))
;;  '(package-selected-packages
;;    (quote
;;     (eglot ccls lsp-mode cmake-font-lock modern-cpp-font-lock cmake-mode lua-mode)))
;;  '(safe-local-variable-values (quote ((TeX-master . t)))))

(defun fill-sentence ()
  (interactive)
  (save-excursion
    (or (eq (point) (point-max)) (forward-char))
    (forward-sentence -1)
    (indent-relative t)
    (let ((beg (point)))
      (forward-sentence)
      (fill-region-as-paragraph beg (point)))))

;;;------------------------------------------------------------------------;;;
;;; Spell check git commit messages.                                       ;;;
;;;------------------------------------------------------------------------;;;
(add-hook 'git-commit-setup-hook 'git-commit-turn-on-flyspell)

;(require 'tex-site)

(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(default ((t (:background "nil")))))
(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(custom-safe-themes
   '("16dd114a84d0aeccc5ad6fd64752a11ea2e841e3853234f19dc02a7b91f5d661" default))
 '(package-selected-packages
   '(dockerfile-mode docker-compose-mode clang-format+ base16-theme magit use-package modern-cpp-font-lock lua-mode eglot cmake-font-lock ccls)))
